#pragma once

#include "litepgo/energy/response/fixed_pattern_response.hpp"

#include <stdexcept>
#include <utility>

namespace litepgo::energy {

// R(x) = A*x + offset. Owns fixed snapshots; A may be rectangular or rank deficient.
// The Jacobian preserves A's stored entries, including explicit zeros. All Hessians are zero.
class AffineResponse final : public FixedPatternResponse {
  public:
    // An empty offset means zero, as in AffineMappedEnergy. Inputs have A.cols() entries.
    explicit AffineResponse(SpMat A, Vec offset = Vec())
        : FixedPatternResponse(A, SpMat(A.cols(), A.cols())), A_(std::move(A)), offset_(std::move(offset)) {
        A_.makeCompressed();
        if (offset_.size() == 0)
            offset_ = Vec::Zero(A_.rows());
        if (offset_.size() != A_.rows() || !A_.coeffs().allFinite() || !offset_.allFinite())
            throw std::invalid_argument("AffineResponse: incompatible or non-finite matrix and offset");
    }

  private:
    static Vec checked_result(Vec result) {
        if (!result.allFinite())
            throw std::domain_error("AffineResponse: non-finite result");
        return result;
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        if (J)
            J->coeffs() = A_.coeffs();
        return checked_result(A_ * x + offset_);
    }
    Vec eval_jvp_impl(const ConstVecRef&, const ConstVecRef& dx) override {
        return checked_result(A_ * dx);
    }
    Vec eval_vjp_impl(const ConstVecRef&, const ConstVecRef& gy) override {
        return checked_result(A_.transpose() * gy);
    }
    void eval_hessian_impl(const ConstVecRef&, const ConstVecRef&, SpMat&) override {}

    SpMat A_;
    Vec offset_;
};

} // namespace litepgo::energy
