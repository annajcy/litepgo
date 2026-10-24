#pragma once

#include "litepgo/fem/interpolated_field.hpp"

namespace litepgo {

// Converts nodal displacement coefficients into I + grad_X u (F or Fp).
// Owns a coefficient snapshot; local DOFs are flattened by row.
template <Basis3D Basis> class DisplacementGradientField final {
  public:
    using FieldValue = Eigen::Matrix3d;
    static constexpr int num_dofs = InterpolatedField<Basis, 3>::num_dofs;

    explicit DisplacementGradientField(const Eigen::Matrix<double, Basis::num_basis_functions, 3>& displacement)
        : interpolation_(displacement) {}

    FieldValue value(const typename VolumeReferenceGeometry<Basis>::PointData& point) const {
        return FieldValue::Identity() + interpolation_.gradient(point);
    }
    static Eigen::Matrix<double, 9, num_dofs>
    jacobian(const typename VolumeReferenceGeometry<Basis>::PointData& point) {
        return InterpolatedField<Basis, 3>::gradient_matrix(point);
    }

  private:
    InterpolatedField<Basis, 3> interpolation_;
};

} // namespace litepgo
