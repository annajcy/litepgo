#pragma once
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include <stdexcept>
namespace litepgo::fem {
// Column-major vec(diag(exp(q))). With two inputs, the third stretch is fixed at one.
class LogDiagonalFpResponse final : public energy::FixedPatternResponse {
  public:
    explicit LogDiagonalFpResponse(int num_stretches = 3) : FixedPatternResponse(pattern(num_stretches)) {}

  private:
    static SpMat pattern(int n) {
        if (n != 2 && n != 3)
            throw std::invalid_argument("LogDiagonalFpResponse: expected two or three stretches");
        SpMat result(9, n);
        for (int c = 0; c < n; ++c)
            result.insert(4 * c, c) = 1.;
        result.makeCompressed();
        return result;
    }
    static Vec stretches(const ConstVecRef& x) {
        Vec y = x.array().exp();
        if (!y.allFinite() || (y.array() <= 0).any())
            throw std::domain_error("LogDiagonalFpResponse: unrepresentable stretches");
        return y;
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        const Vec s = stretches(x);
        Vec y = Vec::Zero(9);
        y[8] = 1.;
        for (int c = 0; c < num_inputs(); ++c) {
            y[4 * c] = s[c];
            if (J)
                J->valuePtr()[c] = s[c];
        }
        return y;
    }
    Vec jvp_at(const ConstVecRef& x, const ConstVecRef& v) override {
        const Vec s = stretches(x);
        Vec y = Vec::Zero(9);
        for (int c = 0; c < num_inputs(); ++c)
            y[4 * c] = s[c] * v[c];
        return y;
    }
    Vec vjp_at(const ConstVecRef& x, const ConstVecRef& w) override {
        Vec y = stretches(x);
        for (int c = 0; c < num_inputs(); ++c)
            y[c] *= w[4 * c];
        return y;
    }
};
} // namespace litepgo::fem
