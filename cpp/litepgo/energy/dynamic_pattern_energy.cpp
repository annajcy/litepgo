#include "litepgo/energy/dynamic_pattern_energy.h"

#include "litepgo/linalg/sparse_pattern.h"

namespace litepgo {

std::optional<SpMat> DynamicPatternEnergy::refresh_pattern(const ConstVecRef& x) {
    SpMat pattern = pattern_of(pattern_at(x));
    if (same_pattern(pattern, hessian_pattern())) {
        return std::nullopt;
    }
    return pattern;
}

} // namespace litepgo
