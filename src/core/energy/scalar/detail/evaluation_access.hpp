#pragma once
#include "litepgo/energy/scalar/energy.hpp"

namespace litepgo::energy::detail {
// Internal borrows remain valid until the evaluation is destroyed.
struct EvaluationAccess {
    // The child must be destroyed before parent, which owns or borrows the shared input.
    static std::shared_ptr<EnergyEvaluation> evaluate_child(const Energy& energy, const EnergyEvaluation& parent) {
        return energy.evaluate_internal(parent.x_, true);
    }
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
