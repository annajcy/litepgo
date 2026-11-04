#pragma once
#include "litepgo/energy/parametric/parametric_energy.hpp"
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/solver/newton.hpp"
#include <memory>

namespace litepgo::solver {
namespace binding {
struct StaticEquilibriumGC;
}

// p -> equilibrium displacement, starting from the same initial state on every call.
// Fixed indices AND values are independent of p. The model defines the total potential,
// including parameter-dependent loads, and must be deterministic.
// JVP/VJP use one directional/adjoint solve without forming the state Jacobian.
// Parameter products delegate to the model; the state Jacobian pattern is still explicit.
// Derivatives require an isolated equilibrium and use the exact, unshifted Hessian.
class StaticEquilibriumResponse final : public energy::FixedPatternResponse {
  public:
    StaticEquilibriumResponse(std::shared_ptr<energy::ParametricEnergy> model, dof::DofReduction fixed, Vec initial,
                              NewtonOptions options);

  private:
    friend struct binding::StaticEquilibriumGC; // Python GC visits and clears the owned model.
    Vec eval_values(const ConstVecRef& p, SpMat* J) const override;
    Vec eval_jvp_values(const ConstVecRef& p, const ConstVecRef& v) const override;
    Vec eval_vjp_values(const ConstVecRef& p, const ConstVecRef& w) const override;
    Vec evaluate(const ConstVecRef& p, SpMat* J, const Vec* direction, bool transpose) const;
    dof::DofReduction fixed_;
    Vec initial_;
    NewtonOptions options_;
    std::shared_ptr<energy::ParametricEnergy> model_;
};
} // namespace litepgo::solver
