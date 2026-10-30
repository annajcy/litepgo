#pragma once

#include "litepgo/contact/contact_options.hpp"
#include "litepgo/energy/dynamic_pattern_energy.hpp"
#include "litepgo/mesh/mesh_data.hpp"
#include <memory>
#include <vector>

namespace litepgo::contact {

// Independent native IPC surface barrier, with unit stiffness. Positions are
// vertex-major xyz. Compose with AffineMappedEnergy for simulation DOFs.
// Uses b(d² - min_distance², dhat * (dhat + 2 min_distance)), and the
// rest-length edge mollifier. Repeated closest features are merged in sorted order
// with multiplicity preserved (the original VF + mollified EE candidate sum).
// Update the dynamic pattern before evaluating a Hessian at a new active set.
//
// Requires nonempty, nondegenerate triangles, no duplicate faces, and every
// vertex used by a face. Planes are normalized and enforce their positive side.
// eval rejects active distances <= min_distance; max_feasible_step additionally
// rejects initial surface intersections. Swept AABBs and additive conservative
// advancement bound the step, returning the last safe bound on exhaustion.
// These are floating-point predicates/ACCD, not formally robust exact arithmetic.
class IPCContactEnergy final : public energy::DynamicPatternEnergy {
  public:
    IPCContactEnergy(const mesh::TriangleMesh& surface, IPCContactOptions options, std::vector<Plane> planes = {});
    ~IPCContactEnergy() override;
    int num_vertices() const {
        return num_dofs() / 3;
    }
    const Vec& rest_positions() const {
        return rest_positions_;
    }
    const IPCContactOptions& options() const {
        return options_;
    }
    int num_active_pairs(const ConstVecRef& x) const;
    double min_active_distance(const ConstVecRef& x) const;

  private:
    struct Impl;
    SpMat pattern_at(const ConstVecRef& x) const override;
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, energy::HessianKind kind) const override;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;
    Vec rest_positions_;
    IPCContactOptions options_;
    std::unique_ptr<const Impl> impl_;
};
} // namespace litepgo::contact
