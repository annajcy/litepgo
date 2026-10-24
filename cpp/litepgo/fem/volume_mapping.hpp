#pragma once

#include "litepgo/fem/direct_local_field.hpp"
#include "litepgo/fem/displacement_gradient_field.hpp"
#include "litepgo/fem/interpolated_value_field.hpp"
#include "litepgo/fem/volume_field_mapping.hpp"
#include "litepgo/fem/volume_local_fields.hpp"
#include "litepgo/fem/volumetric_point.hpp"

namespace litepgo {
namespace detail {

// All volume mappings use the same physical reference coordinates.
template <Basis3D Basis, int NumParameters>
VolumetricPoint<NumParameters> volume_point(const VolumeLocalFields<Basis, NumParameters>& fields,
                                            const typename VolumeReferenceGeometry<Basis>::PointData& reference_point) {
    using VectorField = DisplacementGradientField<Basis>;
    if constexpr (NumParameters > 0) {
        using ParameterField = InterpolatedValueField<Basis, NumParameters>;
        const VolumeFieldInputs<VectorField, VectorField, ParameterField> inputs{
            VectorField(fields.u), VectorField(fields.up), ParameterField(fields.theta)};
        return volume_field_point(inputs, reference_point);
    } else {
        using ParameterField = DirectLocalField<MaterialParameterTypes<0>::Parameters>;
        const VolumeFieldInputs<VectorField, VectorField, ParameterField> inputs{
            VectorField(fields.u), VectorField(fields.up), ParameterField({})};
        return volume_field_point(inputs, reference_point);
    }
}

} // namespace detail

// Shared affine vector-gradient mapping for either fields.u or fields.up.
// The paired point energy selects F or Fp; the other fields are held fixed.
// Local coefficients are flattened by row; point gradients by column.
template <Basis3D Basis, int NumParameters>
    requires requires { typename VolumeLocalFields<Basis, NumParameters>; }
class VolumeVectorGradientMapping final {
  public:
    using Fields = VolumeLocalFields<Basis, NumParameters>;
    using ReferencePoint = typename VolumeReferenceGeometry<Basis>::PointData;
    using Point = VolumetricPoint<NumParameters>;
    static constexpr int num_dofs = 3 * Basis::num_basis_functions;
    static constexpr int point_dofs = 9;
    using PointGradient = Mat3;
    using PointHessian = Eigen::Matrix<double, point_dofs, point_dofs>;
    using Jacobian = Eigen::Matrix<double, point_dofs, num_dofs>;

    Point point_input(const Fields& fields, const ReferencePoint& reference_point) const {
        return detail::volume_point<Basis, NumParameters>(fields, reference_point);
    }

    Jacobian active_jacobian(const ReferencePoint& reference_point) const {
        return DisplacementGradientField<Basis>::jacobian(reference_point);
    }
};

template <Basis3D Basis, int NumParameters> using VolumeFMapping = VolumeVectorGradientMapping<Basis, NumParameters>;

template <Basis3D Basis, int NumParameters> using VolumeFpMapping = VolumeVectorGradientMapping<Basis, NumParameters>;

// Affine mapping with fields.theta active and the other fields fixed.
// Local coefficients are flattened by row; point gradients by column.
template <Basis3D Basis, int NumParameters>
    requires requires { typename VolumeLocalFields<Basis, NumParameters>; }
class VolumeThetaMapping final {
  public:
    using Fields = VolumeLocalFields<Basis, NumParameters>;
    using ReferencePoint = typename VolumeReferenceGeometry<Basis>::PointData;
    using Point = VolumetricPoint<NumParameters>;
    static constexpr int num_dofs = NumParameters * Basis::num_basis_functions;
    static constexpr int point_dofs = NumParameters;
    using PointGradient = typename MaterialParameterTypes<NumParameters>::Parameters;
    using PointHessian = Eigen::Matrix<double, point_dofs, point_dofs>;
    using Jacobian = Eigen::Matrix<double, point_dofs, num_dofs>;

    Point point_input(const Fields& fields, const ReferencePoint& reference_point) const {
        return detail::volume_point<Basis, NumParameters>(fields, reference_point);
    }

    Jacobian active_jacobian(const ReferencePoint& reference_point) const {
        if constexpr (NumParameters > 0) {
            return InterpolatedValueField<Basis, NumParameters>::jacobian(reference_point);
        } else {
            return Jacobian::Zero();
        }
    }
};

} // namespace litepgo
