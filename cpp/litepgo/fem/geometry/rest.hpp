#pragma once

#include "litepgo/fem/geometry/interpolation.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {

// Rest geometry at one parameter location, shared by fields using the same parameter domain.
struct RestPointData {
    Eigen::Vector3d xi;
    Eigen::Vector3d position;
    Eigen::Matrix3d jacobian;         // dX/dxi
    Eigen::Matrix3d inverse_jacobian; // dxi/dX
    double jacobian_determinant;
};

// Validates the reference mapping at xi only; parameters are not clamped.
template <Basis3D B> RestPointData rest_point(const Interpolation<B>& geometry, const Eigen::Vector3d& xi) {
    if (geometry.num_components() != 3)
        throw std::invalid_argument("rest_point requires three geometry components");
    RestPointData point;
    point.xi = xi;
    geometry.value(xi, point.position);
    geometry.gradient(xi, point.jacobian);
    point.jacobian_determinant = point.jacobian.determinant();
    if (!std::isfinite(point.jacobian_determinant) || point.jacobian_determinant <= 0.0)
        throw std::invalid_argument("rest_point: invalid reference mapping");
    point.inverse_jacobian = point.jacobian.inverse();
    if (!point.inverse_jacobian.allFinite())
        throw std::invalid_argument("rest_point: non-finite inverse Jacobian");
    return point;
}

} // namespace litepgo
