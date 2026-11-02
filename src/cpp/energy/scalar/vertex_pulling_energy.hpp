#pragma once

#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"

#include <array>
#include <vector>

namespace litepgo::energy {

// Zero-rest-length springs pulling vertices toward targets:
// E(x) = k/2 sum_i |x[d_i] - t_i|^2, where d_i lists vertex i's three position DOFs
// and t_i occupies targets[3i, 3i+3). Targets use the coordinates of x; when x is a
// displacement, pass anchor minus rest position. The Hessian is the constant
// diagonal k on the pulled DOFs, so both Hessian kinds return it.
class VertexPullingEnergy final : public FixedPatternEnergy {
  public:
    // position_dofs must be distinct and in [0, num_dofs); targets must be finite
    // with three entries per pulled vertex; stiffness must be finite and positive.
    VertexPullingEnergy(int num_dofs, std::vector<std::array<int, 3>> position_dofs, Vec targets, double stiffness);

  private:
    struct Setup {
        int num_dofs;
        std::vector<int> dofs;
        Vec targets;
        double stiffness;
    };
    static Setup checked_setup(int num_dofs, const std::vector<std::array<int, 3>>& position_dofs, Vec targets,
                               double stiffness);
    explicit VertexPullingEnergy(Setup setup);

    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const override;

    std::vector<int> dofs_; // flattened position_dofs, matching targets_
    Vec targets_;
    double stiffness_;
};

} // namespace litepgo::energy
