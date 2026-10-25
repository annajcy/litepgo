#pragma once

#include "litepgo/energy/energy.hpp"

namespace litepgo {

// Base for energies whose Hessian pattern is given at construction and never
// changes, such as quadratic and FEM energies.
class FixedPatternEnergy : public Energy {
  protected:
    FixedPatternEnergy(int num_dofs, const SpMat& pattern) : Energy(num_dofs, pattern, true) {}

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef&) final {
        return std::nullopt;
    }
};

} // namespace litepgo
