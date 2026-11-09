#pragma once
#include "litepgo/fem/assembly/detail/volumetric_fem_force_response.hpp"

namespace litepgo::fem {
// Fixed displacement and geometry. Input: column-major vec(Fp);
// ConstantDofMap shares one block; ElementDofMap assigns a block per element.
// Spatial responses are sampled at construction; no borrowed configuration is retained.
template <Basis3D UBasis, Quadrature3D Q>
using VolumetricFemForceFpResponse = detail::VolumetricFemForceResponse<UBasis, Q, detail::ForceInput::Plastic>;
} // namespace litepgo::fem
