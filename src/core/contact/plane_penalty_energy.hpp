#pragma once

#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"

namespace litepgo::contact {

// One-sided quadratic penalty on packed xyz positions. Axis is 0, 1, or 2.
// At the plane, the Hessian uses the inactive-side value (zero).
class PlanePenaltyEnergy final : public energy::FixedPatternEnergy {
  public:
    PlanePenaltyEnergy(int num_vertices, int axis, double height, double stiffness, bool positive_side = true);

  private:
    PlanePenaltyEnergy(const SpMat& pattern, int axis, double height, double stiffness, bool positive_side);
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, energy::HessianKind) const override;

    int axis_;
    double height_, stiffness_, side_;
};

} // namespace litepgo::contact
