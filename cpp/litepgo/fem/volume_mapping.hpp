#pragma once

#include "litepgo/fem/interpolated_field.hpp"
#include "litepgo/fem/volume_local_fields.hpp"
#include "litepgo/fem/volumetric_point.hpp"

namespace litepgo {
namespace detail {

// All volume mappings use the same physical reference coordinates.
template <Basis3D Basis, int NumParameters>
VolumetricPoint<NumParameters> volume_point(const VolumeLocalFields<Basis, NumParameters>& fields,
                                            const typename VolumeReferenceGeometry<Basis>::PointData& reference_point) {
    const InterpolatedField<Basis, 3> u(fields.u), up(fields.up);
    VolumetricPoint<NumParameters> result;
    result.F = Mat3::Identity() + u.gradient(reference_point);
    result.Fp = Mat3::Identity() + up.gradient(reference_point);
    if constexpr (NumParameters > 0) {
        const InterpolatedField<Basis, NumParameters> parameters(fields.theta);
        result.theta = parameters.value(reference_point);
    }
    return result;
}

} // namespace detail

// Affine mapping with fields.u active and the other fields fixed.
// Local coefficients are flattened by row; point gradients by column.
template <Basis3D Basis, int NumParameters>
    requires requires { typename VolumeLocalFields<Basis, NumParameters>; }
class VolumeFMapping final {
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
        return InterpolatedField<Basis, 3>::gradient_matrix(reference_point);
    }
};

// Affine mapping with fields.up active and the other fields fixed.
// Local coefficients are flattened by row; point gradients by column.
template <Basis3D Basis, int NumParameters>
    requires requires { typename VolumeLocalFields<Basis, NumParameters>; }
class VolumeFpMapping final {
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
        return InterpolatedField<Basis, 3>::gradient_matrix(reference_point);
    }
};

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
            return InterpolatedField<Basis, NumParameters>::value_matrix(reference_point);
        } else {
            return Jacobian::Zero();
        }
    }
};

} // namespace litepgo
