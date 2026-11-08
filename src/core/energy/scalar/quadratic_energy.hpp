#pragma once

#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"

namespace litepgo::energy {

// E(x) = 1/2 x^T A x + b^T x with a symmetric A (pattern and values). A is not
// projected: both Hessian kinds return it, so with an indefinite A the "projected"
// Hessian is not PSD.
class QuadraticEnergy final : public FixedPatternEnergy {
  public:
    // b may be empty, meaning zero.
    explicit QuadraticEnergy(SpMat A, Vec b = Vec());

  private:
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const override;

    SpMat A_;
    Vec b_;
};

} // namespace litepgo::energy
