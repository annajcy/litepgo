#pragma once
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include <stdexcept>
namespace litepgo::energy {
// Positive physical parameters from unconstrained logarithms.
class ExpResponse final : public FixedPatternResponse {
  public:
    explicit ExpResponse(int size) : FixedPatternResponse(pattern(size), pattern(size)) {}

  private:
    static SpMat pattern(int size) {
        if (size <= 0)
            throw std::invalid_argument("ExpResponse: positive size required");
        SpMat result(size, size);
        result.setIdentity();
        return result;
    }
    static Vec values(const ConstVecRef& x) {
        Vec y = x.array().exp();
        if (!y.allFinite() || (y.array() <= 0).any())
            throw std::domain_error("ExpResponse: unrepresentable exponential");
        return y;
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) const override {
        Vec y = values(x);
        if (J)
            for (int i = 0; i < num_inputs(); ++i)
                J->valuePtr()[i] = y[i];
        return y;
    }
    void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) const override {
        H.coeffs() = values(x).cwiseProduct(weights);
    }
    Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& v) override {
        return values(x).cwiseProduct(v);
    }
    Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& w) override {
        return values(x).cwiseProduct(w);
    }
};
} // namespace litepgo::energy
