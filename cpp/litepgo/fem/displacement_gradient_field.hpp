#pragma once

#include "litepgo/fem/interpolated_field.hpp"

namespace litepgo {

// Converts nodal displacement coefficients into I + grad_X u (F or Fp).
// Owns a coefficient snapshot; local DOFs are flattened by row.
template <Basis3D Basis> class DisplacementGradientField final {
  public:
    using Interpolation = InterpolatedField<Basis, 3>;
    using Coefficients = typename Interpolation::LocalCoefficients;
    using ReferencePoint = typename Interpolation::ReferencePoint;
    using Value = Eigen::Matrix3d;
    static constexpr int num_dofs = Interpolation::num_dofs;
    using Jacobian = typename Interpolation::GradientMatrix;

    explicit DisplacementGradientField(const Coefficients& displacement) : interpolation_(displacement) {}

    Value value(const ReferencePoint& point) const {
        return Value::Identity() + interpolation_.gradient(point);
    }
    static Jacobian jacobian(const ReferencePoint& point) {
        return Interpolation::gradient_matrix(point);
    }

  private:
    Interpolation interpolation_;
};

} // namespace litepgo
