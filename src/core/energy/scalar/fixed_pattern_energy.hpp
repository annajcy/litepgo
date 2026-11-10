#pragma once
#include "litepgo/energy/scalar/detail/sparse_energy.hpp"
namespace litepgo::energy {
class FixedPatternEnergy : public detail::SparseEnergy {
  protected:
    explicit FixedPatternEnergy(const SpMat& pattern)
        : SparseEnergy(static_cast<int>(pattern.rows())),
          hessian_pattern_(checked_hessian_pattern(pattern, num_dofs())) {}

  protected:
    SpMat make_hessian_pattern(const ConstVecRef&) const final {
        return hessian_pattern_;
    }

  private:
    SpMat hessian_pattern_;
};
} // namespace litepgo::energy
