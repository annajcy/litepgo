#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/broad_phase/sweep_and_prune.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <unordered_map>

namespace litepgo::contact {
namespace {
using Cell = std::array<std::int64_t, 3>;
struct CellHash {
    std::size_t operator()(const Cell& cell) const {
        std::size_t hash = 0;
        for (auto coordinate : cell)
            hash ^=
                std::hash<std::int64_t>{}(coordinate) + std::size_t(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2);
        return hash;
    }
};
struct CellRange {
    Cell lo, hi;
};
struct Record {
    Cell cell;
    std::size_t box;
};
constexpr std::size_t max_records = 1 << 20;
} // namespace

SpatialHash::SpatialHash(std::optional<double> cell_size) : cell_size_(cell_size) {
    if (cell_size && (!std::isfinite(*cell_size) || *cell_size <= 0))
        throw std::invalid_argument("SpatialHash: cell_size must be finite and positive, or unset");
}

std::vector<BoxPair> SpatialHash::detect(const BroadPhaseQuery& query) const {
    bool finite = true;
    // Validate the complete query before any fallback, including boxes after an infinity.
    for (const auto& box : query.boxes) {
        if (box.lo.array().isNaN().any() || box.hi.array().isNaN().any() || (box.lo.array() > box.hi.array()).any() ||
            (box.kind != PrimitiveKind::Vertex && box.kind != PrimitiveKind::Edge && box.kind != PrimitiveKind::Face))
            throw std::invalid_argument("SpatialHash: invalid AABB");
        finite = finite && box.lo.allFinite() && box.hi.allFinite();
    }
    const auto fallback = [&] { return SweepAndPrune{}.detect(query); };
    if (!finite)
        return fallback();
    if (query.boxes.empty())
        return {};
    double h = cell_size_.value_or(0);
    if (!cell_size_) {
        std::vector<double> extents;
        for (const auto& box : query.boxes) {
            if (box.kind == PrimitiveKind::Vertex)
                continue;
            const double extent = (box.hi - box.lo).maxCoeff();
            if (!std::isfinite(extent))
                return fallback();
            if (extent > 0)
                extents.push_back(extent);
        }
        if (extents.empty())
            return fallback();
        const auto middle = extents.begin() + static_cast<std::ptrdiff_t>(extents.size() / 2);
        std::nth_element(extents.begin(), middle, extents.end());
        h = *middle;
    }
    std::vector<CellRange> ranges(query.boxes.size());
    std::vector<std::size_t> offsets(query.boxes.size() + 1, 0);
    for (std::size_t i = 0; i < query.boxes.size(); ++i) {
        std::size_t count = 1;
        for (int k = 0; k < 3; ++k) {
            const double lo = std::floor(query.boxes[i].lo[k] / h);
            const double hi = std::floor(query.boxes[i].hi[k] / h);
            // INT64_MAX rounds to 2^63 in double, so the upper test must be strict.
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo < -0x1p63 || hi >= 0x1p63)
                return fallback();
            ranges[i].lo[k] = static_cast<std::int64_t>(lo);
            ranges[i].hi[k] = static_cast<std::int64_t>(hi);
            const auto difference =
                static_cast<std::uint64_t>(ranges[i].hi[k]) - static_cast<std::uint64_t>(ranges[i].lo[k]);
            if (difference >= max_records)
                return fallback();
            const auto width = static_cast<std::size_t>(difference) + 1;
            if (count > max_records / width)
                return fallback();
            count *= width;
        }
        if (offsets[i] > max_records - count)
            return fallback();
        offsets[i + 1] = offsets[i] + count;
    }
    std::vector<Record> records(offsets.back());
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, ranges.size()), [&](const auto& work) {
        for (std::size_t i = work.begin(); i < work.end(); ++i) {
            const auto& range = ranges[i];
            auto next = offsets[i];
            for (auto x = range.lo[0];; ++x) {
                for (auto y = range.lo[1];; ++y) {
                    for (auto z = range.lo[2];; ++z) {
                        records[next++] = {{x, y, z}, i};
                        if (z == range.hi[2])
                            break;
                    }
                    if (y == range.hi[1])
                        break;
                }
                if (x == range.hi[0])
                    break;
            }
        }
    });
    // Full int3 equality resolves hash collisions; no compressed key can alias cells.
    using Grid = std::unordered_map<Cell, std::vector<std::size_t>, CellHash>;
    Grid grid;
    for (const auto& record : records)
        grid[record.cell].push_back(record.box);
    std::vector<const Grid::value_type*> buckets;
    buckets.reserve(grid.size());
    for (const auto& bucket : grid)
        buckets.push_back(&bucket);
    tbb::enumerable_thread_specific<std::vector<BoxPair>> output;
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, buckets.size()), [&](const auto& work) {
        auto& pairs = output.local();
        for (std::size_t b = work.begin(); b < work.end(); ++b) {
            const auto& [cell, ids] = *buckets[b];
            for (std::size_t i = 0; i < ids.size(); ++i)
                for (std::size_t j = i + 1; j < ids.size(); ++j) {
                    const auto ai = ids[i], bi = ids[j];
                    const auto& a = query.boxes[ai];
                    const auto& b_box = query.boxes[bi];
                    if (!query.allowed_pairs.allows(a.kind, b_box.kind))
                        continue;
                    const Cell owner{std::max(ranges[ai].lo[0], ranges[bi].lo[0]),
                                     std::max(ranges[ai].lo[1], ranges[bi].lo[1]),
                                     std::max(ranges[ai].lo[2], ranges[bi].lo[2])};
                    // The lowest shared cell owns the pair, even when both boxes span many cells.
                    if (cell != owner)
                        continue;
                    if ((a.lo.array() > b_box.hi.array()).any() || (b_box.lo.array() > a.hi.array()).any())
                        continue;
                    pairs.push_back({ai, bi});
                }
        }
    });
    std::vector<BoxPair> result;
    for (const auto& pairs : output)
        result.insert(result.end(), pairs.begin(), pairs.end());
    return result;
}
} // namespace litepgo::contact
