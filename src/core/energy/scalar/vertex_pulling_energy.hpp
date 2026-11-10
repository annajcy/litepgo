#pragma once

#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"

#include <array>
#include <vector>

namespace litepgo::energy {

// Zero-rest-length springs pulling vertices toward targets:
// E(x) = k/2 sum_i |x[d_i] - t_i|^2, where d_i lists vertex i's three position DOFs
// and t_i occupies targets[3i, 3i+3). Targets use the coordinates of x; when x is a
// displacement, pass anchor minus rest position. The Hessian is the constant
// diagonal k on the pulled DOFs, so both Hessian methods return it.
class VertexPullingEnergy final : public FixedPatternEnergy {
  public:
    // position_dofs must be distinct and in [0, num_dofs); targets must be finite
    // with three entries per pulled vertex; stiffness must be finite and positive.
    VertexPullingEnergy(int num_dofs, std::vector<std::array<int, 3>> position_dofs, Vec targets, double stiffness);

  private:
    energy::EnergyValues eval_projected_values(const ConstVecRef&, energy::EnergyEvaluation& exact) const override {
        SpMat H = exact.hessian();
        return {exact.value(), exact.gradient(), std::move(H)};
    }

    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian) const override;

    std::vector<std::array<int, 3>> position_dofs_;
    Vec targets_;
    double stiffness_;
};

} // namespace litepgo::energy
