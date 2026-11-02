#pragma once
#include "litepgo/fem/volumetric_fem_force_response.hpp"

namespace litepgo::fem {
// Fixed displacement and geometry. Input: column-major vec(Fp);
// ConstantDofMap shares one block; ElementDofMap assigns a block per element.
// Fields are owned by value; borrowed dependencies must outlive the response and stay fixed.
template <class Material, Basis3D UBasis, Quadrature3D Q, std::derived_from<Field3D> XField,
          std::derived_from<Field3D> ThetaField>
using VolumetricFemForceFpResponse =
    detail::VolumetricFemForceResponse<Material, UBasis, Q, XField, ThetaField, detail::ForceInput::Plastic>;
} // namespace litepgo::fem
