#pragma once
#include "litepgo/fem/material/volumetric/volumetric_force_local_response.hpp"

namespace litepgo::fem {
// Positive F-coefficient force density; Jacobian wrt Fp coefficients.
// The other input fields and dF/dcoeffs are fixed. Borrowed fields must outlive the response.
template <class Material>
using VolumetricFpForceLocalResponse3D = detail::VolumetricForceLocalResponse3D<Material, detail::ForceInput::Plastic>;
} // namespace litepgo::fem
