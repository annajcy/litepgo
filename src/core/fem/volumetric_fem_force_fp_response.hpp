#pragma once
#include "litepgo/fem/volumetric_fem_force_response.hpp"

namespace litepgo::fem {
// Fixed displacement and geometry. Input: column-major vec(Fp);
// ConstantDofMap shares one block; ElementDofMap assigns a block per element.
// Spatial responses are sampled at construction; no borrowed configuration is retained.
template <class Material, Basis3D UBasis, Quadrature3D Q>
using VolumetricFemForceFpResponse =
    detail::VolumetricFemForceResponse<Material, UBasis, Q, detail::ForceInput::Plastic>;
} // namespace litepgo::fem
