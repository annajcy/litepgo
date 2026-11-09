#pragma once
#include "litepgo/energy/response/response.hpp"
#include "litepgo/solver/newton.hpp"
#include <functional>
#include <memory>

namespace litepgo::solver {
// p -> equilibrium displacement, starting from the same initial state for each evaluation.
// Fixed indices AND values are independent of p. Callbacks must be deterministic.
// make_energy(p) supplies the total potential; make_response(u) maps p to its state gradient.
// Callers guarantee consistency, including parameter-dependent loads.
// JVP/VJP use one directional/adjoint solve without forming the state Jacobian.
// Each evaluation retains its equilibrium and prepares exact stiffness on first derivative use.
// Derivatives require an isolated equilibrium and use the exact, unshifted Hessian.
class StaticEquilibriumResponse : public energy::Response {
  public:
    using MakeEnergy = std::function<std::shared_ptr<energy::Energy>(const Vec&)>;
    using MakeResponse = std::function<std::shared_ptr<energy::Response>(const Vec&)>;
    StaticEquilibriumResponse(int num_parameters, dof::DofReduction fixed, Vec initial, NewtonOptions options,
                              MakeEnergy make_energy, MakeResponse make_response);

    int num_inputs() const noexcept override {
        return parameters_;
    }
    int num_outputs() const noexcept override {
        return static_cast<int>(initial_.size());
    }

  private:
    std::shared_ptr<energy::ResponseEvaluation> make_evaluation(const ConstVecRef& p) override;
    int parameters_;
    dof::DofReduction fixed_;
    Vec initial_;
    NewtonOptions options_;
    MakeEnergy make_energy_;
    MakeResponse make_response_;
};
} // namespace litepgo::solver
