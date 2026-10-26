#pragma once

#include "litepgo/fem/geometry/interpolation.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {

// Rest geometry at one parameter location, shared by fields using the same parameter domain.
struct RestPointData {
    Eigen::Vector3d xi;
    Eigen::Vector3d X;
    Eigen::Matrix3d dX_dxi;
    Eigen::Matrix3d dxi_dX;
    double det_dX_dxi;
};

// Validates the reference mapping at xi only; parameters are not clamped.
template <Basis3D B> RestPointData rest_point(const Interpolation<B>& geometry, const Eigen::Vector3d& xi) {
    if (geometry.num_components() != 3)
        throw std::invalid_argument("rest_point requires three geometry components");
    RestPointData point;
    point.xi = xi;
    point.X = geometry.value(xi);
    point.dX_dxi = geometry.dvalue_dxi(xi);
    point.det_dX_dxi = point.dX_dxi.determinant();
    if (!std::isfinite(point.det_dX_dxi) || point.det_dX_dxi <= 0.0)
        throw std::invalid_argument("rest_point: invalid reference mapping");
    point.dxi_dX = point.dX_dxi.inverse();
    if (!point.dxi_dX.allFinite())
        throw std::invalid_argument("rest_point: non-finite inverse Jacobian");
    return point;
}

} // namespace litepgo
