#pragma once
#include "litepgo/common/types.hpp"
#include <optional>
namespace litepgo::energy {
// Selects the exact second derivative or an implementation-provided solver approximation.
enum class HessianKind {
    // The exact Hessian.
    Exact,
    // Implementation-provided solver approximation, not necessarily PSD.
    // Mesh/contact energies project local terms; composites combine child approximations.
    // QuadraticEnergy returns Exact even for an indefinite matrix.
    Projected,
};

struct EnergyEvaluation {
    double value;
    std::optional<Vec> gradient;
    std::optional<SpMat> hessian;
};
// Mathematical scalar function. Results own their storage; implementations protect shared workspaces.
class Energy {
  public:
    virtual ~Energy() = default;
    Energy(const Energy&) = delete;
    Energy& operator=(const Energy&) = delete;
    int num_dofs() const noexcept {
        return num_dofs_;
    }
    EnergyEvaluation eval(const ConstVecRef& x, bool gradient = false,
                          std::optional<HessianKind> hessian = std::nullopt) const;
    double eval_value(const ConstVecRef& x) const;
    Vec eval_gradient(const ConstVecRef& x) const;
    SpMat eval_hessian(const ConstVecRef& x, HessianKind kind = HessianKind::Exact) const;
    // Conservative bound for every t in [0, alpha], including alpha; x must be feasible.
    // Not necessarily maximal and does not guarantee representable floating-point results.
    double max_feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const;

  protected:
    explicit Energy(int num_dofs);
    virtual EnergyEvaluation eval_impl(const ConstVecRef& x, bool gradient,
                                       std::optional<HessianKind> hessian) const = 0;
    virtual double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const;

  private:
    void check_size(const ConstVecRef& x, const char* caller) const;
    int num_dofs_;
};
} // namespace litepgo::energy
