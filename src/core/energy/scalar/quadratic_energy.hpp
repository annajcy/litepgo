#pragma once

#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"
#include <mutex>

namespace litepgo::energy {

// E(x) = 1/2 x^T A x + b^T x with symmetric A. Its PSD projection is cached.
class QuadraticEnergy final : public FixedPatternEnergy {
  public:
    // b may be empty, meaning zero.
    explicit QuadraticEnergy(SpMat A, Vec b = Vec());

  private:
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian) const override;

    EnergyValues eval_projected_values(const ConstVecRef& x, EnergyEvaluation&) const override;
    mutable std::once_flag projection_once_;
    mutable SpMat projected_;
    SpMat A_;
    Vec b_;
};

} // namespace litepgo::energy
