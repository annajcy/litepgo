#pragma once
#include "litepgo/energy/response/response.hpp"

namespace litepgo::energy {
// Base for responses whose supported derivative structures never change.
// Derived classes implement eval_values and overwrite requested numerical values.
class FixedPatternResponse : public Response {
  protected:
    explicit FixedPatternResponse(const SpMat& pattern, std::optional<SpMat> hessian_pattern = std::nullopt)
        : Response(pattern, true, std::move(hessian_pattern)) {}

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef&) final {
        return std::nullopt;
    }
};
} // namespace litepgo::energy
