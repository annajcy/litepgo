#include "litepgo/energy/scalar/dynamic_pattern_energy.hpp"

#include "litepgo/linalg/sparse_pattern.hpp"

namespace litepgo::energy {

std::optional<SpMat> DynamicPatternEnergy::refresh_pattern(const ConstVecRef& x) {
    SpMat pattern = linalg::pattern_of(pattern_at(x));
    if (linalg::same_pattern(pattern, hessian_pattern())) {
        return std::nullopt;
    }
    return pattern;
}

} // namespace litepgo::energy
