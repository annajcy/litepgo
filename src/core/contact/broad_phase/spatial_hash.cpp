#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/spatial/spatial_hash.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace litepgo::contact {
SpatialHash::SpatialHash(std::optional<double> cell_size, std::size_t grid_memory_budget)
    : cell_size_(cell_size), grid_memory_budget_(grid_memory_budget) {
    if (!grid_memory_budget)
        throw std::invalid_argument("SpatialHash: grid memory budget must be positive");
    if (cell_size && (!std::isfinite(*cell_size) || *cell_size <= 0))
        throw std::invalid_argument("SpatialHash: cell_size must be finite and positive, or unset");
}

std::vector<BoxPair> SpatialHash::detect(const BroadPhaseQuery& query) const {
    std::vector<spatial::AABB> boxes;
    std::vector<double> extents;
    for (const auto& b : query.boxes) {
        if (b.kind != PrimitiveKind::Vertex && b.kind != PrimitiveKind::Edge && b.kind != PrimitiveKind::Face)
            throw std::invalid_argument("SpatialHash: invalid primitive kind");
        boxes.push_back({b.lo, b.hi});
        const double e = (b.hi - b.lo).maxCoeff();
        if (b.kind != PrimitiveKind::Vertex && std::isfinite(e) && e > 0)
            extents.push_back(e);
    }
    auto size = cell_size_;
    if (!size && !extents.empty()) {
        auto mid = extents.begin() + static_cast<std::ptrdiff_t>(extents.size() / 2);
        std::nth_element(extents.begin(), mid, extents.end());
        size = *mid;
    }
    return spatial::SpatialHash(boxes, size, grid_memory_budget_).overlapping_pairs([&](auto i, auto j) {
        return query.allowed_pairs.allows(query.boxes[i].kind, query.boxes[j].kind);
    });
}
} // namespace litepgo::contact
