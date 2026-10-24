#pragma once

#include "litepgo/fem/material_parameter_types.hpp"

namespace litepgo {

// Owns the pointwise inputs; no geometry, interpolation, or DOF layout.
template <int NumParameters>
    requires(NumParameters >= 0)
struct VolumetricPoint {
    Mat3 F;
    Mat3 Fp;
    typename MaterialParameterTypes<NumParameters>::Parameters theta;
};

} // namespace litepgo
