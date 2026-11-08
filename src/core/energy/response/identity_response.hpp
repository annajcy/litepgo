#pragma once

#include "litepgo/energy/response/fixed_pattern_response.hpp"

#include <stdexcept>

namespace litepgo::energy {

// R(x) = x with direct products and a zero weighted Hessian.
class IdentityResponse final : public FixedPatternResponse {
  public:
    explicit IdentityResponse(int size) : IdentityResponse(identity_pattern(size)) {}

  private:
    explicit IdentityResponse(const SpMat& pattern)
        : FixedPatternResponse(pattern, SpMat(pattern.cols(), pattern.cols())) {}

    static SpMat identity_pattern(int size) {
        if (size < 0)
            throw std::invalid_argument("IdentityResponse: negative size");
        SpMat pattern(size, size);
        if (size > 0)
            pattern.setIdentity();
        return pattern;
    }

    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        if (J)
            J->coeffs().setOnes();
        return x;
    }
    Vec eval_jvp_impl(const ConstVecRef&, const ConstVecRef& dx) override {
        return dx;
    }
    Vec eval_vjp_impl(const ConstVecRef&, const ConstVecRef& weights) override {
        return weights;
    }
    void eval_hessian_impl(const ConstVecRef&, const ConstVecRef&, SpMat&) override {}
};

} // namespace litepgo::energy
