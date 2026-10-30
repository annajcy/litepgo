#pragma once

#include "litepgo/core/types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tuple>
#include <vector>

namespace litepgo::contact::detail {

struct Candidates {
    std::vector<std::array<int, 4>> vf, ee, ef;
};

// Positions/topology are validated by IPCContactEnergy. Swept boxes contain the
// entire linear trajectory; inflation and outward rounding retain touching pairs.
inline Candidates broad_phase(const ConstVecRef& x, const Vec* dx, double inflation,
                              const std::vector<std::array<int, 2>>& edges,
                              const std::vector<std::array<int, 3>>& faces, bool intersections = false) {
    using Ids = std::array<int, 4>;
    using V = Eigen::Vector3d;
    struct Box {
        V lo, hi;
        int kind, id;
    }; // vertex=0, edge=1, face=2
    const int nv = static_cast<int>(x.size() / 3);
    const auto shared = [](const Ids& a, int na, const Ids& b, int nb) {
        for (int i = 0; i < na; ++i)
            for (int j = 0; j < nb; ++j)
                if (a[i] == b[j])
                    return true;
        return false;
    };
    std::vector<Box> boxes;
    boxes.reserve(nv + edges.size() + faces.size());
    auto add = [&](Ids ids, int n, int kind, int id) {
        Box b{V::Constant(std::numeric_limits<double>::infinity()),
              V::Constant(-std::numeric_limits<double>::infinity()), kind, id};
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
        add({v, -1, -1, -1}, 1, 0, v);
    for (int e = 0; e < static_cast<int>(edges.size()); ++e)
        add({edges[e][0], edges[e][1], -1, -1}, 2, 1, e);
    for (int f = 0; f < static_cast<int>(faces.size()); ++f)
        add({faces[f][0], faces[f][1], faces[f][2], -1}, 3, 2, f);
    V lo = V::Constant(INFINITY), hi = V::Constant(-INFINITY);
    for (const auto& box : boxes) {
        lo = lo.cwiseMin(box.lo);
        hi = hi.cwiseMax(box.hi);
    }
    Eigen::Index axis = 0;
    if (!boxes.empty())
        (hi - lo).maxCoeff(&axis);
    std::sort(boxes.begin(), boxes.end(), [axis](const Box& a, const Box& b) {
        return std::tuple{a.lo[axis], a.kind, a.id} < std::tuple{b.lo[axis], b.kind, b.id};
    });
    tbb::enumerable_thread_specific<Candidates> buffers;
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, boxes.size(), 64), [&](const auto& range) {
        auto& result = buffers.local();
        for (std::size_t i = range.begin(); i < range.end(); ++i)
            for (std::size_t j = i + 1; j < boxes.size() && boxes[j].lo[axis] <= boxes[i].hi[axis]; ++j) {
                const Box* a = &boxes[i];
                const Box* b = &boxes[j];
                if (a->kind > b->kind)
                    std::swap(a, b);
                if (!((a->kind == 0 && b->kind == 2) || (a->kind == 1 && b->kind == 1) ||
                      (intersections && a->kind == 1 && b->kind == 2)))
                    continue;
                if ((a->lo.array() > b->hi.array()).any() || (b->lo.array() > a->hi.array()).any())
                    continue;
                if (a->kind == 0) {
                    auto f = faces[b->id];
                    if (a->id == f[0] || a->id == f[1] || a->id == f[2])
                        continue;
                    result.vf.push_back({a->id, f[0], f[1], f[2]});
                } else if (b->kind == 1) {
                    int ai = std::min(a->id, b->id), bi = std::max(a->id, b->id);
                    auto e = edges[ai], f = edges[bi];
                    if (shared({e[0], e[1], -1, -1}, 2, {f[0], f[1], -1, -1}, 2))
                        continue;
                    result.ee.push_back({ai, bi, -1, -1});
                } else {
                    auto e = edges[a->id];
                    auto f = faces[b->id];
                    if (shared({e[0], e[1], -1, -1}, 2, {f[0], f[1], f[2], -1}, 3))
                        continue;
                    result.ef.push_back({a->id, b->id, -1, -1});
                }
            }
    });
    Candidates result;
    for (const auto& buffer : buffers) {
        result.vf.insert(result.vf.end(), buffer.vf.begin(), buffer.vf.end());
        result.ee.insert(result.ee.end(), buffer.ee.begin(), buffer.ee.end());
        result.ef.insert(result.ef.end(), buffer.ef.begin(), buffer.ef.end());
    }
    // Canonical ordering makes downstream reductions independent of task scheduling.
    for (auto* list : {&result.vf, &result.ee, &result.ef})
        std::sort(list->begin(), list->end());
    return result;
}

} // namespace litepgo::contact::detail
