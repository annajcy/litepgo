#pragma once
#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <stdexcept>
namespace litepgo::energy::detail {
class SparseEnergy : public Energy {
  protected:
    using Energy::Energy;
    static SpMat checked_hessian_pattern(SpMat p, int n) {
        if (p.rows() != n || p.cols() != n)
            throw std::invalid_argument("Energy: invalid pattern dimensions");
        p.makeCompressed();
        p.coeffs().setZero();
        if (!linalg::is_canonical(p) || !linalg::same_pattern(p, SpMat(p.transpose())))
            throw std::invalid_argument("Energy: invalid sparse pattern");
        return p;
    }
    virtual SpMat make_hessian_pattern(const ConstVecRef& x) const = 0;
    // Overwrite every requested entry while preserving the prepared structure.
    virtual double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian) const = 0;

  private:
    EnergyValues eval_impl(const ConstVecRef& x, bool gradient, bool hessian) const final {
        EnergyValues result{0, std::nullopt, std::nullopt};
        if (gradient)
            result.gradient.emplace(num_dofs());
        if (hessian)
            result.hessian = make_hessian_pattern(x);
#ifndef NDEBUG
        const auto pattern = result.hessian;
#endif
        result.value = eval_values(x, gradient ? &*result.gradient : nullptr, hessian ? &*result.hessian : nullptr);
#ifndef NDEBUG
        if (hessian && !linalg::same_pattern(*result.hessian, *pattern))
            throw std::logic_error("Energy: Hessian structure changed during evaluation");
#endif
        return result;
    }
};
} // namespace litepgo::energy::detail
