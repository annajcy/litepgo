#pragma once

#include "litepgo/contact/contact_options.hpp"
#include "litepgo/contact/contact_pair_weights.hpp"
#include "litepgo/contact/contact_snapshot.hpp"
#include "litepgo/energy/scalar/dynamic_pattern_energy.hpp"
#include "litepgo/mesh/mesh_data.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace litepgo::contact {

// IPC's clamped log barrier for self-contact on a triangle surface
// [Li et al. 2020], evaluated by IPC Toolkit. x holds the surface vertices'
// positions in vertex-major xyz order. The stiffness is 1: scale the energy by its
// weight in a WeightedSumEnergy.
// use_area_weighting enables fixed reference-area collision weights.
// use_physical_barrier selects IPC Toolkit's physical normalization, with
// pressure units for the external stiffness when area weighting is enabled.
// Optional group multipliers act before closest-feature merging. Zero excludes
// candidates from the active set and CCD, while initial whole-surface intersection
// validation remains unconditional. Configuration is owned by value.
//
// The active set at x is every vertex-face, edge-edge (mollified near parallel),
// edge-vertex and vertex-vertex pair closer than min_distance + dhat;
// primitives sharing a vertex never pair. The Hessian pattern holds a dense 3x3
// block for every two vertices of an active pair, so it changes with the active
// set. Each Hessian call prepares that structure automatically.
//
// A state is feasible when the surface does not intersect itself, every pair is
// with a positive group multiplier is farther apart than min_distance.
// eval() throws std::domain_error for an active
// pair within min_distance.
// max_feasible_step() first checks all of x's feasibility (std::domain_error), then
// returns the earliest conservative impact time over Tight-Inclusion CCD queries of
// IPC Toolkit's candidates, which also keeps min_distance.
//
// Pairs are evaluated in a fixed order, so results are deterministic.
class IPCTKContactEnergy final : public energy::DynamicPatternEnergy {
  public:
    // surface may contain several disconnected components; it must not be empty.
    // Its positions are the rest geometry, which IPC Toolkit uses for the edge-edge
    // mollifier threshold.
    // dhat must be finite and positive, min_distance finite and non-negative.
    IPCTKContactEnergy(const mesh::TriangleMesh& surface, IPCTKContactOptions options,
                       std::optional<ContactPairWeights> pair_weights = std::nullopt);
    ~IPCTKContactEnergy() override;

    int num_vertices() const {
        return num_dofs() / 3;
    }
    // The surface's rest positions, in the order of x.
    const Vec& rest_positions() const {
        return rest_positions_;
    }
    const IPCTKContactOptions& options() const {
        return options_;
    }

    // Diagnostics at x, independent of other evaluations: the number of active
    // pairs, and the smallest distance among them (+infinity when there are none).
    int num_active_pairs(const ConstVecRef& x) const;
    double min_active_distance(const ConstVecRef& x) const;
    // Full owned snapshot; does not change evaluation results or solver state.
    ContactSnapshot contact_snapshot(const ConstVecRef& x) const;

  private:
    struct Impl;

    SpMat hessian_pattern_at(const ConstVecRef& x) const override;
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian) const override;
    energy::EnergyValues eval_projected_values(const ConstVecRef& x, energy::EnergyEvaluation&) const override;
    double assemble_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, bool project) const;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;

    Vec rest_positions_;
    IPCTKContactOptions options_;
    std::unique_ptr<const Impl> impl_;
};

} // namespace litepgo::contact
