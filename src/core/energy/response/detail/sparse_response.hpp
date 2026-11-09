#pragma once
#include "litepgo/energy/response/response.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <stdexcept>
namespace litepgo::energy::detail {
// Internal adapter: all derivative storage belongs to the current call.
class SparseResponse : public Response {
  protected:
    static SpMat checked_pattern(SpMat p, Eigen::Index rows, Eigen::Index cols, bool symmetric = false) {
        if (p.rows() != rows || p.cols() != cols)
            throw std::invalid_argument("Response: invalid pattern dimensions");
        p.makeCompressed();
        p.coeffs().setZero();
        if (!linalg::is_canonical(p) || (symmetric && !linalg::same_pattern(p, SpMat(p.transpose()))))
            throw std::invalid_argument("Response: invalid sparse pattern");
        return p;
    }
    // Implementations return a normalized, validated structure.
    virtual SpMat make_jacobian_pattern(const ConstVecRef& x) const = 0;
    virtual SpMat make_hessian_pattern(const ConstVecRef& x) const = 0;
    virtual Vec eval_impl(const ConstVecRef& x, SpMat* jacobian) = 0;
    virtual void eval_hessian_impl(const ConstVecRef&, const ConstVecRef&, SpMat&) {
        throw std::logic_error("Response does not implement Hessian evaluation");
    }

  private:
    Vec value_at(const ConstVecRef& x) final {
        return eval_impl(x, nullptr);
    }
    std::pair<Vec, SpMat> value_jacobian_at(const ConstVecRef& x) final {
        SpMat J = make_jacobian_pattern(x);
#ifndef NDEBUG
        const SpMat pattern = J;
#endif
        Vec value = eval_impl(x, &J);
#ifndef NDEBUG
        if (!linalg::same_pattern(J, pattern))
            throw std::logic_error("Response: Jacobian structure changed during evaluation");
#endif
        return {std::move(value), std::move(J)};
    }
    SpMat hessian_at(const ConstVecRef& x, const ConstVecRef& weights) final {
        SpMat H = make_hessian_pattern(x);
#ifndef NDEBUG
        const SpMat pattern = H;
#endif
        eval_hessian_impl(x, weights, H);
#ifndef NDEBUG
        if (!linalg::same_pattern(H, pattern))
            throw std::logic_error("Response: Hessian structure changed during evaluation");
#endif
        return H;
    }
};
} // namespace litepgo::energy::detail
