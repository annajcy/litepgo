#pragma once
#include "litepgo/fem/material/volumetric/volumetric_force_local_response.hpp"

namespace litepgo::fem {
// Positive F-coefficient force density; Jacobian wrt theta coefficients.
// The other input fields and dF/dcoeffs are fixed. Borrowed fields must outlive the response.
template <class Material>
using VolumetricMaterialForceLocalResponse3D =
    detail::VolumetricForceLocalResponse3D<Material, detail::ForceInput::Material>;
} // namespace litepgo::fem
