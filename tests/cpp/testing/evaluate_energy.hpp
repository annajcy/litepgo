#pragma once
#include "litepgo/energy/scalar/energy.hpp"
namespace litepgo::testing {
inline double evaluate_energy(const energy::Energy& energy, const ConstVecRef& x, Vec* g, SpMat* H,
                              bool project = false) {
    auto result = energy.evaluate(x);
    if (H)
        *H = project ? result->projected_hessian() : result->hessian();
    if (g)
        *g = result->gradient();
    return result->value();
}
} // namespace litepgo::testing
