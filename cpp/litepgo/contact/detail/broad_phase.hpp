#pragma once

#include "litepgo/contact/broad_phase.hpp"
#include "litepgo/core/types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace litepgo::contact::detail {

struct Candidates {
    std::vector<std::array<int, 4>> vf, ee, ef;
};

// Positions/topology are validated by IPCContactEnergy. Swept boxes contain the
// entire linear trajectory; inflation and outward rounding retain touching pairs.
inline Candidates broad_phase(const ConstVecRef& x, const Vec* dx, double inflation,
                              const std::vector<std::array<int, 2>>& edges,
                              const std::vector<std::array<int, 3>>& faces, bool intersections,
                              const BroadPhase& detector) {
    using Ids = std::array<int, 4>;
    using V = Eigen::Vector3d;
    const int nv = static_cast<int>(x.size() / 3);
    const auto shared = [](const Ids& a, int na, const Ids& b, int nb) {
        for (int i = 0; i < na; ++i)
            for (int j = 0; j < nb; ++j)
                if (a[i] == b[j])
                    return true;
        return false;
    };
    std::vector<AABB> boxes;
    boxes.reserve(nv + edges.size() + faces.size());
    auto add = [&](Ids ids, int n, PrimitiveKind kind) {
        AABB b{V::Constant(std::numeric_limits<double>::infinity()),
               V::Constant(-std::numeric_limits<double>::infinity()), kind};
        for (int i = 0; i < n; ++i) {
            V p = x.segment<3>(3 * ids[i]);
            b.lo = b.lo.cwiseMin(p);
            b.hi = b.hi.cwiseMax(p);
            if (dx) {
                p += dx->segment<3>(3 * ids[i]);
                b.lo = b.lo.cwiseMin(p);
                b.hi = b.hi.cwiseMax(p);
            }
        }
        b.lo.array() -= inflation;
        b.hi.array() += inflation;
        if (!b.lo.allFinite() || !b.hi.allFinite())
            throw std::domain_error("IPCContactEnergy: unrepresentable bounding box");
        // Outward one-ULP rounding avoids dropping touching boxes after inflation.
        for (int k = 0; k < 3; ++k) {
            b.lo[k] = std::nextafter(b.lo[k], -INFINITY);
            b.hi[k] = std::nextafter(b.hi[k], INFINITY);
        }
        boxes.push_back(b);
    };
    for (int v = 0; v < nv; ++v)
        add({v, -1, -1, -1}, 1, PrimitiveKind::Vertex);
    for (int e = 0; e < static_cast<int>(edges.size()); ++e)
        add({edges[e][0], edges[e][1], -1, -1}, 2, PrimitiveKind::Edge);
    for (int f = 0; f < static_cast<int>(faces.size()); ++f)
        add({faces[f][0], faces[f][1], faces[f][2], -1}, 3, PrimitiveKind::Face);
    const PairKindMask mask{true, true, intersections};
    Candidates result;
    for (auto indices : detector.detect({boxes, mask})) {
        if (indices[0] >= boxes.size() || indices[1] >= boxes.size())
            throw std::invalid_argument("IPCContactEnergy: broad phase returned an invalid box index");
        auto [a, b] = indices;
        if (a == b)
            continue;
        if (boxes[a].kind > boxes[b].kind)
            std::swap(a, b);
        if (!mask.allows(boxes[a].kind, boxes[b].kind) || (boxes[a].lo.array() > boxes[b].hi.array()).any() ||
            (boxes[b].lo.array() > boxes[a].hi.array()).any())
            continue;
        // Input order is vertices, edges, faces; backends return those original indices.
        if (boxes[a].kind == PrimitiveKind::Vertex) {
            const int v = static_cast<int>(a);
            const auto f = faces[b - nv - edges.size()];
            if (v != f[0] && v != f[1] && v != f[2])
                result.vf.push_back({v, f[0], f[1], f[2]});
        } else if (boxes[b].kind == PrimitiveKind::Edge) {
            const int ai = static_cast<int>(std::min(a, b) - nv), bi = static_cast<int>(std::max(a, b) - nv);
            const auto e = edges[ai], f = edges[bi];
            if (!shared({e[0], e[1], -1, -1}, 2, {f[0], f[1], -1, -1}, 2))
                result.ee.push_back({ai, bi, -1, -1});
        } else {
            const int ei = static_cast<int>(a - nv), fi = static_cast<int>(b - nv - edges.size());
            const auto e = edges[ei];
            const auto f = faces[fi];
            if (!shared({e[0], e[1], -1, -1}, 2, {f[0], f[1], f[2], -1}, 3))
                result.ef.push_back({ei, fi, -1, -1});
        }
    }
    // Normalize backend order and duplicates before contact multiplicities are formed.
    for (auto* list : {&result.vf, &result.ee, &result.ef}) {
        std::sort(list->begin(), list->end());
        list->erase(std::unique(list->begin(), list->end()), list->end());
    }
    return result;
}
} // namespace litepgo::contact::detail
