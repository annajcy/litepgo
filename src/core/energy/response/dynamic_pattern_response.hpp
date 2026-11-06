#pragma once
#include "litepgo/energy/response/response.hpp"

namespace litepgo::energy {
// Fixed dimensions, state-dependent structure. Pattern hooks describe structural supersets,
// ignoring stored values. Ordinary evaluation prepares them; *_into requires update_patterns(x).
class DynamicPatternResponse : public Response {
  protected:
    explicit DynamicPatternResponse(const SpMat& initial_pattern,
                                    std::optional<SpMat> initial_hessian_pattern = std::nullopt)
        : Response(initial_pattern, false, std::move(initial_hessian_pattern)) {}
    virtual SpMat jacobian_pattern_at(const ConstVecRef& x) const override = 0;
};
} // namespace litepgo::energy
