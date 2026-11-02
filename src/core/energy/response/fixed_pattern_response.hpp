#pragma once
#include "litepgo/energy/response/response.hpp"

namespace litepgo::energy {
// Base for responses whose Jacobian structure never changes.
// Derived classes implement eval_values and overwrite requested numerical values.
class FixedPatternResponse : public Response {
  protected:
    explicit FixedPatternResponse(const SpMat& pattern) : Response(pattern, true) {}

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef&) final {
        return std::nullopt;
    }
};
} // namespace litepgo::energy
