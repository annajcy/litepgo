#pragma once
#include "litepgo/energy/scalar/energy.hpp"

namespace litepgo::energy::detail {
// Internal borrows remain valid until the evaluation is destroyed.
struct EvaluationAccess {
    static const Vec& gradient(EnergyEvaluation& e) {
        if (!e.gradient_)
            e.save(e.compute(true, false), true, false, false);
        return *e.gradient_;
    }
    static const SpMat& hessian(EnergyEvaluation& e) {
        if (!e.hessian_)
            e.save(e.compute(true, true), true, true, false);
        return *e.hessian_;
    }
    static const SpMat& projected_hessian(EnergyEvaluation& e) {
        if (!e.projected_)
            e.save(e.compute_projected(), true, true, true);
        return *e.projected_;
    }
};
} // namespace litepgo::energy::detail
