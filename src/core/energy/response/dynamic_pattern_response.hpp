#pragma once
#include "litepgo/energy/response/response.hpp"

namespace litepgo::energy {
// Fixed input/output dimensions, state-dependent Jacobian structure.
// pattern_at describes all potentially active entries at x; stored values are ignored.
// update_pattern validates and installs it only when the structure differs.
// Value-only evaluation is independent of the current pattern. Derivative evaluation
// must reject nonzero entries outside it; unused slots must be overwritten with zero.
class DynamicPatternResponse : public Response {
  protected:
    explicit DynamicPatternResponse(const SpMat& initial_pattern,
                                    std::optional<SpMat> initial_hessian_pattern = std::nullopt)
        : Response(initial_pattern, false, std::move(initial_hessian_pattern)) {}

    // Override when the union of component Hessian structures changes with x.
    virtual SpMat hessian_pattern_at(const ConstVecRef&) const {
        return hessian_pattern();
    }

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef& x) final {
        return pattern_at(x);
    }
    std::optional<SpMat> refresh_hessian_pattern(const ConstVecRef& x) final {
        return hessian_pattern_at(x);
    }
    virtual SpMat pattern_at(const ConstVecRef& x) const = 0;
};
} // namespace litepgo::energy
