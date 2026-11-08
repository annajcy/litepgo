#pragma once
#include "litepgo/energy/scalar/energy.hpp"
namespace litepgo::testing {
// Unpack independently owned results for shared derivative assertions.
inline double evaluate_energy(const energy::Energy& energy, const ConstVecRef& x, Vec* g, SpMat* H,
                              energy::HessianKind kind = energy::HessianKind::Exact) {
    auto result = energy.eval(x, g != nullptr, H ? std::optional(kind) : std::nullopt);
    if (g)
        *g = std::move(*result.gradient);
    if (H)
        *H = std::move(*result.hessian);
    return result.value;
}
} // namespace litepgo::testing
