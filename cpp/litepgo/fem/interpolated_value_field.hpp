#pragma once

#include "litepgo/fem/interpolated_field.hpp"

namespace litepgo {

// Takes the interpolated value, with no spatial derivative or identity offset.
// Owns a coefficient snapshot; local DOFs are flattened by row.
template <Basis3D Basis, int Components>
    requires(Components > 0)
class InterpolatedValueField final {
  public:
    using FieldValue = Eigen::Matrix<double, Components, 1>;
    static constexpr int num_dofs = InterpolatedField<Basis, Components>::num_dofs;

    explicit InterpolatedValueField(const Eigen::Matrix<double, Basis::num_basis_functions, Components>& coefficients)
        : interpolation_(coefficients) {}

    FieldValue value(const typename VolumeReferenceGeometry<Basis>::PointData& point) const {
        return interpolation_.value(point);
    }
    static Eigen::Matrix<double, Components, num_dofs>
    jacobian(const typename VolumeReferenceGeometry<Basis>::PointData& point) {
        return InterpolatedField<Basis, Components>::value_matrix(point);
    }

  private:
    InterpolatedField<Basis, Components> interpolation_;
};

} // namespace litepgo
