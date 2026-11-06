#pragma once
#include "litepgo/energy/response/response.hpp"

namespace litepgo::energy {
// Both supported derivative structures stay fixed. Subclasses implement eval_impl and optional derivative hooks.
class FixedPatternResponse : public Response {
  protected:
    explicit FixedPatternResponse(const SpMat& pattern, std::optional<SpMat> hessian_pattern = std::nullopt)
        : Response(pattern, true, std::move(hessian_pattern)) {}
};
} // namespace litepgo::energy
