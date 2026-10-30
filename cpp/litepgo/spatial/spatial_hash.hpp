#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <unordered_map>
#include <vector>

namespace litepgo::spatial {
struct AABB {
    Eigen::Vector3d lo, hi;
};
using BoxPair = std::array<std::size_t, 2>;

// Owns input boxes. IDs are input indices. Const queries may run concurrently.
class SpatialHash {
    using Cell = std::array<std::int64_t, 3>;
    struct Hash {
        std::size_t operator()(const Cell& cell) const {
            std::size_t h = 0;
            for (auto c : cell)
                h ^= std::hash<std::int64_t>{}(c) + std::size_t(0x9e3779b97f4a7c15ULL) + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct Range {
        Cell lo, hi;
        std::size_t count = 1;
    };
    static constexpr std::size_t limit = 1 << 20;
    std::vector<AABB> boxes_;
    std::vector<Range> ranges_;
    std::unordered_map<Cell, std::vector<std::size_t>, Hash> grid_;
    double h_ = 1;
    static void validate(const AABB& b) {
        if (b.lo.array().isNaN().any() || b.hi.array().isNaN().any() || (b.lo.array() > b.hi.array()).any())
            throw std::invalid_argument("SpatialHash: invalid AABB");
    }
    static bool overlaps(const AABB& a, const AABB& b) {
        return !(a.lo.array() > b.hi.array()).any() && !(b.lo.array() > a.hi.array()).any();
    }
    Range range(const AABB& b) const {
        Range r;
        for (int k = 0; k < 3; ++k) {
            double lo = std::floor(b.lo[k] / h_), hi = std::floor(b.hi[k] / h_);
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo < -0x1p63 || hi >= 0x1p63)
                throw std::overflow_error("SpatialHash: grid coordinates are not representable");
            r.lo[k] = static_cast<std::int64_t>(lo);
            r.hi[k] = static_cast<std::int64_t>(hi);
            auto d = static_cast<std::uint64_t>(r.hi[k]) - static_cast<std::uint64_t>(r.lo[k]);
            if (d >= limit || r.count > limit / (d + 1))
                throw std::length_error("SpatialHash: box or query exceeds the cell record limit");
            r.count *= static_cast<std::size_t>(d + 1);
        }
        return r;
    }
    template <class F> static void cells(const Range& r, F f) {
        for (auto x = r.lo[0];; ++x) {
            for (auto y = r.lo[1];; ++y) {
                for (auto z = r.lo[2];; ++z) {
                    f(Cell{x, y, z});
                    if (z == r.hi[2])
                        break;
                }
                if (y == r.hi[1])
                    break;
            }
            if (x == r.hi[0])
                break;
        }
    }

  public:
    explicit SpatialHash(std::span<const AABB> boxes, std::optional<double> cell_size = std::nullopt)
        : boxes_(boxes.begin(), boxes.end()) {
        if (cell_size && (!std::isfinite(*cell_size) || *cell_size <= 0))
            throw std::invalid_argument("SpatialHash: invalid cell size");
        std::vector<double> extents;
        for (const auto& b : boxes_) {
            validate(b);
            double e = (b.hi - b.lo).maxCoeff();
            if (std::isfinite(e) && e > 0)
                extents.push_back(e);
        }
        if (cell_size)
            h_ = *cell_size;
        else if (!extents.empty()) {
            auto mid = extents.begin() + static_cast<std::ptrdiff_t>(extents.size() / 2);
            std::nth_element(extents.begin(), mid, extents.end());
            h_ = *mid;
        }
        std::size_t count = 0;
        for (const auto& b : boxes_) {
            auto r = range(b);
            if (count > limit - r.count)
                throw std::length_error("SpatialHash: total cell records exceed the limit");
            count += r.count;
            ranges_.push_back(r);
        }
        struct Record {
            Cell cell;
            std::size_t box;
        };
        std::vector<std::size_t> offsets(ranges_.size() + 1, 0);
        for (std::size_t i = 0; i < ranges_.size(); ++i)
            offsets[i + 1] = offsets[i] + ranges_[i].count;
        std::vector<Record> records(count);
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, ranges_.size()), [&](const auto& work) {
            for (auto i = work.begin(); i < work.end(); ++i) {
                auto next = offsets[i];
                cells(ranges_[i], [&](Cell c) { records[next++] = {c, i}; });
            }
        });
        for (const auto& record : records)
            grid_[record.cell].push_back(record.box);
    }
    // Inclusive intersection, unique original IDs; result order is unspecified.
    std::vector<std::size_t> query(const AABB& region) const {
        validate(region);
        std::vector<std::size_t> ids;
        auto r = range(region);
        cells(r, [&](Cell c) {
            auto it = grid_.find(c);
            if (it != grid_.end())
                for (auto i : it->second)
                    if (overlaps(region, boxes_[i]))
                        ids.push_back(i);
        });
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
    }
    // Filter must support concurrent calls; result order is unspecified. Each overlapping accepted pair appears once, i
    // < j.
    template <class PairFilter>
        requires std::predicate<PairFilter&, std::size_t, std::size_t>
    std::vector<BoxPair> overlapping_pairs(PairFilter&& accept) const {
        tbb::enumerable_thread_specific<std::vector<BoxPair>> buffers;
        auto emit = [&](std::vector<BoxPair>& pairs, std::size_t i, std::size_t j) {
            if (i > j)
                std::swap(i, j);
            if (accept(i, j) && overlaps(boxes_[i], boxes_[j]))
                pairs.push_back({i, j});
        };
        std::vector<const decltype(grid_)::value_type*> buckets;
        buckets.reserve(grid_.size());
        for (const auto& bucket : grid_)
            buckets.push_back(&bucket);
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, buckets.size()), [&](const auto& work) {
            auto& pairs = buffers.local();
            for (auto bucket = work.begin(); bucket < work.end(); ++bucket) {
                const auto& [cell, ids] = *buckets[bucket];
                for (std::size_t a = 0; a < ids.size(); ++a)
                    for (std::size_t b = a + 1; b < ids.size(); ++b) {
                        const auto i = ids[a], j = ids[b];
                        Cell owner;
                        for (int k = 0; k < 3; ++k)
                            owner[k] = std::max(ranges_[i].lo[k], ranges_[j].lo[k]);
                        if (cell == owner)
                            emit(pairs, i, j);
                    }
            }
        });
        std::vector<BoxPair> pairs;
        for (const auto& buffer : buffers)
            pairs.insert(pairs.end(), buffer.begin(), buffer.end());
        return pairs;
    }
};
} // namespace litepgo::spatial
