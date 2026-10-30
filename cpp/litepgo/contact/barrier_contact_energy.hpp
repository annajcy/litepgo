#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/energy/dynamic_pattern_energy.hpp"
#include "litepgo/mesh/mesh_data.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace litepgo::contact {

// A static infinite plane through point with the given normal (nonzero, normalized
// on construction). Vertices must stay on the side the normal points to.
struct Plane {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
};

struct BarrierContactOptions {
    // Required: the barrier acts on primitive pairs closer than min_distance + dhat.
    double dhat = std::numeric_limits<double>::quiet_NaN();
    // Distances must stay above it: a thickness, or 0 for touching primitives.
    double min_distance = 0.0;
    // Tight-Inclusion CCD's time-of-impact tolerance, and the iteration budget after
    // which it returns its current conservative bound: a smaller budget can shorten
    // steps, never make them unsafe. Nearly parallel edges in contact are its slowest case.
    double ccd_tolerance = 1e-6;
    long ccd_max_iterations = 100'000;
};

// IPC's clamped log barrier between the triangles of a surface and static planes
// [Li et al. 2020], evaluated by IPC Toolkit. x holds the surface vertices'
// positions in vertex-major xyz order. The stiffness is 1: scale the energy by its
// weight in a WeightedEnergySum.
//
// The active set at x is every vertex-face, edge-edge (mollified near parallel),
// edge-vertex, vertex-vertex and plane-vertex pair closer than min_distance + dhat;
// primitives sharing a vertex never pair. The Hessian pattern holds a dense 3x3
// block for every two vertices of an active pair, so it changes with the active
// set; an active pair outside the current pattern is a std::logic_error.
//
// A state is feasible when the surface does not intersect itself, every pair is
// farther apart than min_distance, and every vertex is more than min_distance in
// front of every plane. eval() throws std::domain_error at an active pair within
// min_distance or a vertex not that far in front of a plane; it does not search
// for intersections between distant primitives.
// max_feasible_step() first checks all of x's feasibility (std::domain_error), then
// returns the earliest conservative impact time over Tight-Inclusion CCD queries of
// IPC Toolkit's candidates, which also keeps min_distance, capped
// for each vertex at the same fraction (0.8) of its reach to min_distance from a plane.
//
// Pairs are evaluated in a fixed order, so results are deterministic.
class BarrierContactEnergy final : public energy::DynamicPatternEnergy {
  public:
    // surface may contain several disconnected components; it must not be empty.
    // Its positions are the rest geometry, which IPC Toolkit uses for the edge-edge
    // mollifier threshold.
    // dhat must be finite and positive, min_distance finite and non-negative.
    BarrierContactEnergy(const mesh::TriangleMesh& surface, BarrierContactOptions options,
                         std::vector<Plane> planes = {});
    ~BarrierContactEnergy() override;

    int num_vertices() const {
        return num_dofs() / 3;
    }
    // The surface's rest positions, in the order of x.
    const Vec& rest_positions() const {
        return rest_positions_;
    }
    const BarrierContactOptions& options() const {
        return options_;
    }

    // Diagnostics at x, independent of the current pattern: the number of active
    // pairs, and the smallest distance among them (+infinity when there are none).
    int num_active_pairs(const ConstVecRef& x) const;
    double min_active_distance(const ConstVecRef& x) const;

  private:
    struct Impl;

    SpMat pattern_at(const ConstVecRef& x) const override;
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, energy::HessianKind kind) const override;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;

    Vec rest_positions_;
    BarrierContactOptions options_;
    std::unique_ptr<const Impl> impl_;
};

} // namespace litepgo::contact
