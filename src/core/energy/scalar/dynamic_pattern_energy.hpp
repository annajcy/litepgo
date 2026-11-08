#pragma once
#include "litepgo/energy/scalar/detail/sparse_energy.hpp"
namespace litepgo::energy {
class DynamicPatternEnergy : public detail::SparseEnergy {
  protected:
    explicit DynamicPatternEnergy(int num_dofs) : SparseEnergy(num_dofs) {}
    virtual SpMat hessian_pattern_at(const ConstVecRef& x) const = 0;

  private:
    SpMat make_hessian_pattern(const ConstVecRef& x) const final {
        return checked_hessian_pattern(hessian_pattern_at(x), num_dofs());
    }
};
} // namespace litepgo::energy
