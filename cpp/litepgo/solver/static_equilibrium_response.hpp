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
// make_energy supplies the total potential; partial supplies d(grad_u potential)/dp
// at fixed u, including parameter-dependent loads. Both may throw on invalid p.
// Derivatives require an isolated equilibrium and use the exact, unshifted Hessian.
class StaticEquilibriumResponse final : public energy::FixedPatternResponse {
  public:
    using MakeEnergy = std::function<std::shared_ptr<energy::Energy>(const Vec&)>;
    using Partial = std::function<SpMat(const Vec&, const Vec&)>;
    StaticEquilibriumResponse(int num_parameters, dof::DofReduction fixed, Vec initial, NewtonOptions options,
                              MakeEnergy make_energy, Partial partial);

  private:
    friend struct binding::StaticEquilibriumGC; // Python GC visits and clears the owned callbacks.
    Vec eval_values(const ConstVecRef& p, SpMat* J) const override;
    dof::DofReduction fixed_;
    Vec initial_;
    NewtonOptions options_;
    MakeEnergy make_energy_;
    Partial partial_;
};
} // namespace litepgo::solver
