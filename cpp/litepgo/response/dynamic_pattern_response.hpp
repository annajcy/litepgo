#pragma once
#include "litepgo/response/response.hpp"

namespace litepgo::response {
// Fixed input/output dimensions, state-dependent Jacobian structure.
// pattern_at describes all potentially active entries at x; stored values are ignored.
// update_pattern validates and installs it only when the structure differs.
// Value-only evaluation is independent of the current pattern. Derivative evaluation
// must reject nonzero entries outside it; unused slots must be overwritten with zero.
class DynamicPatternResponse : public Response {
  protected:
    explicit DynamicPatternResponse(const SpMat& initial_pattern) : Response(initial_pattern, false) {}

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef& x) final {
        return pattern_at(x);
    }
    virtual SpMat pattern_at(const ConstVecRef& x) const = 0;
};
} // namespace litepgo::response
