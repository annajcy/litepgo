#pragma once

#include "litepgo/energy/energy.hpp"

namespace litepgo::energy {

// Base for energies whose Hessian pattern depends on the state, such as contact,
// whose pattern follows the active pairs. A subclass states the pattern it needs
// at x; update_pattern() installs it only when it differs from the current one.
// The subclass's evaluation must reject (std::logic_error) an active entry at x
// that falls outside the current pattern.
class DynamicPatternEnergy : public Energy {
  protected:
    DynamicPatternEnergy(int num_dofs, const SpMat& initial_pattern) : Energy(num_dofs, initial_pattern, false) {}

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef& x) final;
    virtual SpMat pattern_at(const ConstVecRef& x) const = 0;
};

} // namespace litepgo::energy
