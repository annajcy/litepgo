#include "litepgo/contact/sweep_and_prune.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tuple>

namespace litepgo::contact {
std::vector<BoxPair> SweepAndPrune::detect(const BroadPhaseQuery& query) const {
    using V = Eigen::Vector3d;
    struct Box {
        V lo, hi;
        PrimitiveKind kind;
        std::size_t index;
    };
    std::vector<Box> boxes;
    boxes.reserve(query.boxes.size());
    for (std::size_t i = 0; i < query.boxes.size(); ++i) {
        const auto& box = query.boxes[i];
        if (box.lo.array().isNaN().any() || box.hi.array().isNaN().any() || (box.lo.array() > box.hi.array()).any() ||
            (box.kind != PrimitiveKind::Vertex && box.kind != PrimitiveKind::Edge && box.kind != PrimitiveKind::Face))
            throw std::invalid_argument("SweepAndPrune: invalid AABB");
        boxes.push_back({box.lo, box.hi, box.kind, i});
    }
    V lo = V::Constant(INFINITY), hi = V::Constant(-INFINITY);
    for (const auto& box : boxes) {
        lo = lo.cwiseMin(box.lo);
        hi = hi.cwiseMax(box.hi);
    }
    Eigen::Index axis = 0;
    if (!boxes.empty()) {
        V extent;
        for (int k = 0; k < 3; ++k)
            extent[k] = hi[k] == lo[k] ? 0 : hi[k] - lo[k]; // Equal infinities have zero extent, not NaN.
        extent.maxCoeff(&axis);
    }
    std::sort(boxes.begin(), boxes.end(), [axis](const Box& a, const Box& b) {
        return std::tuple{a.lo[axis], a.kind, a.index} < std::tuple{b.lo[axis], b.kind, b.index};
    });
    tbb::enumerable_thread_specific<std::vector<BoxPair>> buffers;
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, boxes.size(), 64), [&](const auto& range) {
        auto& result = buffers.local();
        for (std::size_t i = range.begin(); i < range.end(); ++i)
            for (std::size_t j = i + 1; j < boxes.size() && boxes[j].lo[axis] <= boxes[i].hi[axis]; ++j) {
                const auto& a = boxes[i];
                const auto& b = boxes[j];
                if (!query.allowed_pairs.allows(a.kind, b.kind))
                    continue;
                if ((a.lo.array() > b.hi.array()).any() || (b.lo.array() > a.hi.array()).any())
                    continue;
                result.push_back({a.index, b.index});
            }
    });
    std::vector<BoxPair> result;
    for (const auto& buffer : buffers)
        result.insert(result.end(), buffer.begin(), buffer.end());
    return result;
}
} // namespace litepgo::contact
