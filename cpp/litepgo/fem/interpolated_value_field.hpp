#pragma once

#include "litepgo/fem/interpolated_field.hpp"

namespace litepgo {

// Takes the interpolated value, with no spatial derivative or identity offset.
// Owns a coefficient snapshot; local DOFs are flattened by row.
template <Basis3D Basis, int Components>
    requires(Components > 0)
class InterpolatedValueField final {
  public:
    using Interpolation = InterpolatedField<Basis, Components>;
    using Coefficients = typename Interpolation::LocalCoefficients;
    using ReferencePoint = typename Interpolation::ReferencePoint;
    using Value = typename Interpolation::Value;
    static constexpr int num_dofs = Interpolation::num_dofs;
    using Jacobian = typename Interpolation::ValueMatrix;

    explicit InterpolatedValueField(const Coefficients& coefficients) : interpolation_(coefficients) {}

    Value value(const ReferencePoint& point) const {
        return interpolation_.value(point);
    }
    static Jacobian jacobian(const ReferencePoint& point) {
        return Interpolation::value_matrix(point);
    }

  private:
    Interpolation interpolation_;
};

} // namespace litepgo
