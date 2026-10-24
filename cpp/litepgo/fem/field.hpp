#pragma once

#include "litepgo/fem/field_value.hpp"

namespace litepgo {

// A local field snapshot evaluated at a reference geometry point.
template <class T, class GeometryPoint>
concept Field = requires(const T& field, const GeometryPoint& point) {
    typename T::FieldValue;
    requires FixedFieldValue<typename T::FieldValue>;
    { field.value(point) } -> std::same_as<typename T::FieldValue>;
};

// At fixed geometry, vec(value) = jacobian(point) * local_dofs + offset(point).
// vec uses column-major order. Affineness is a mathematical contract, not checked by C++.
template <class T, class GeometryPoint>
concept AffineField = Field<T, GeometryPoint> && requires(const T& field, const GeometryPoint& point) {
    requires(T::num_dofs > 0);
    { field.jacobian(point) } -> std::same_as<Eigen::Matrix<double, T::FieldValue::SizeAtCompileTime, T::num_dofs>>;
};

} // namespace litepgo
