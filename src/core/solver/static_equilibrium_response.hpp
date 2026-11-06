#pragma once
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/solver/newton.hpp"
#include <functional>
#include <memory>

namespace litepgo::solver {
namespace binding {
struct StaticEquilibriumGC;
}

// p -> equilibrium displacement, starting from the same initial state on every call.
// Fixed indices AND values are independent of p. Callbacks must be deterministic.
// make_energy(p) supplies the total potential; make_response(u) maps p to its state gradient.
// Callers guarantee consistency, including parameter-dependent loads.
// JVP/VJP use one directional/adjoint solve without forming the state Jacobian.
// Parameter products delegate to the response; the declared Jacobian pattern remains explicit.
// Derivatives require an isolated equilibrium and use the exact, unshifted Hessian.
class StaticEquilibriumResponse final : public energy::FixedPatternResponse {
  public:
    using MakeEnergy = std::function<std::shared_ptr<energy::Energy>(const Vec&)>;
    using MakeResponse = std::function<std::shared_ptr<energy::Response>(const Vec&)>;
    StaticEquilibriumResponse(int num_parameters, dof::DofReduction fixed, Vec initial, NewtonOptions options,
                              MakeEnergy make_energy, MakeResponse make_response);

  private:
    friend struct binding::StaticEquilibriumGC; // Python GC visits and clears the owned callbacks.
    Vec eval_impl(const ConstVecRef& p, SpMat* J) override;
    Vec eval_jvp_impl(const ConstVecRef& p, const ConstVecRef& v) override;
    Vec eval_vjp_impl(const ConstVecRef& p, const ConstVecRef& w) override;
    Vec evaluate(const ConstVecRef& p, SpMat* J, const Vec* direction, bool transpose) const;
    dof::DofReduction fixed_;
    Vec initial_;
    NewtonOptions options_;
    MakeEnergy make_energy_;
    MakeResponse make_response_;
};
} // namespace litepgo::solver
