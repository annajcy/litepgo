#pragma once
#include "litepgo/energy/response/response.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include <atomic>
#include <memory>

namespace litepgo::energy {
// E(u,p). Models are shared-owned and immutable apart from scratch storage.
// Views copy their fixed argument and own independent sparsity patterns.
// All views serialize access to the model; nested/overlapping calls are rejected.
class ParametricEnergy : public std::enable_shared_from_this<ParametricEnergy> {
  public:
    virtual ~ParametricEnergy() = default;
    ParametricEnergy(const ParametricEnergy&) = delete;
    ParametricEnergy& operator=(const ParametricEnergy&) = delete;
    int num_dofs() const {
        return static_cast<int>(state_pattern_.rows());
    }
    int num_parameters() const {
        return static_cast<int>(parameter_pattern_.cols());
    }
    std::shared_ptr<Energy> energy_at(const ConstVecRef& p);
    std::shared_ptr<Response> force_at(const ConstVecRef& u);

  protected:
    // Patterns describe K=d²E/du² and B=d(grad_u E)/dp. Dynamic hooks return
    // complete patterns for the supplied pair, never shared mutable "current" patterns.
    ParametricEnergy(const SpMat& state_pattern, const SpMat& parameter_pattern, bool fixed_pattern = true);
    virtual double eval_state_values(const ConstVecRef& u, const ConstVecRef& p, Vec* g, SpMat* K,
                                     HessianKind kind) const = 0;
    virtual void eval_parameter_jacobian(const ConstVecRef& u, const ConstVecRef& p, SpMat& B) const = 0;
    virtual Vec eval_parameter_jvp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& v) const;
    virtual Vec eval_parameter_vjp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& w) const;
    virtual SpMat state_pattern_at(const ConstVecRef& u, const ConstVecRef& p) const;
    virtual SpMat parameter_pattern_at(const ConstVecRef& u, const ConstVecRef& p) const;
    virtual double eval_feasible_step(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& du) const;

  private:
    class EnergyView;
    class ForceView;
    SpMat parameter_jacobian(const ConstVecRef& u, const ConstVecRef& p) const;
    SpMat state_pattern_, parameter_pattern_;
    bool fixed_pattern_;
    mutable std::atomic<bool> busy_ = false;
};
} // namespace litepgo::energy
