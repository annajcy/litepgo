#pragma once

#include "litepgo/energy/response/fixed_pattern_response.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace litepgo::energy {

// R(x) = value. Owns a snapshot of value and has zero derivatives.
class ConstantResponse final : public FixedPatternResponse {
  public:
    ConstantResponse(int num_inputs, Vec value)
        : ConstantResponse(zero_pattern(num_inputs, value.size()), std::move(value)) {}

  private:
    ConstantResponse(const SpMat& pattern, Vec&& value)
        : FixedPatternResponse(pattern, SpMat(pattern.cols(), pattern.cols())), value_(std::move(value)) {
        if (!value_.allFinite())
            throw std::invalid_argument("ConstantResponse: non-finite value");
    }

    static SpMat zero_pattern(int num_inputs, Eigen::Index num_outputs) {
        if (num_inputs < 0 || num_outputs > std::numeric_limits<int>::max())
            throw std::invalid_argument("ConstantResponse: invalid dimensions");
        return SpMat(num_outputs, num_inputs);
    }

    Vec eval_impl(const ConstVecRef&, SpMat*) override {
        return value_;
    }
    Vec jvp_at(const ConstVecRef&, const ConstVecRef&) override {
        return Vec::Zero(num_outputs());
    }
    Vec vjp_at(const ConstVecRef&, const ConstVecRef&) override {
        return Vec::Zero(num_inputs());
    }
    void eval_hessian_impl(const ConstVecRef&, const ConstVecRef&, SpMat&) override {}

    Vec value_;
};

} // namespace litepgo::energy
