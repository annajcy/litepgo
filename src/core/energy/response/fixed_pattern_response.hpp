#pragma once
#include "litepgo/energy/response/detail/sparse_response.hpp"
#include <optional>
namespace litepgo::energy {
class FixedPatternResponse : public detail::SparseResponse {
  public:
    int num_inputs() const noexcept final {
        return static_cast<int>(jacobian_pattern_.cols());
    }
    int num_outputs() const noexcept final {
        return static_cast<int>(jacobian_pattern_.rows());
    }
    bool has_hessian() const override {
        return hessian_pattern_.has_value();
    }

  protected:
    explicit FixedPatternResponse(const SpMat& pattern, std::optional<SpMat> hessian_pattern = std::nullopt)
        : jacobian_pattern_(checked_pattern(pattern, pattern.rows(), pattern.cols())) {
        if (hessian_pattern)
            hessian_pattern_ = checked_pattern(*hessian_pattern, num_inputs(), num_inputs(), true);
    }

  private:
    SpMat make_jacobian_pattern(const ConstVecRef&) const final {
        return jacobian_pattern_;
    }
    SpMat make_hessian_pattern(const ConstVecRef&) const final {
        if (!hessian_pattern_)
            throw std::logic_error("Response does not support Hessians");
        return *hessian_pattern_;
    }
    SpMat jacobian_pattern_;
    std::optional<SpMat> hessian_pattern_;
};
} // namespace litepgo::energy
