#pragma once

#include "litepgo/fem/geometry/rest.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

// Geometry and quadrature must use the same parameter domain.
// Integrates the density and requested DOF derivatives in one pass. Outputs are
// independently nullable, resized and overwritten; they must not alias energy data.
// Unsupported derivative requests throw std::logic_error before any outputs change.
template <Quadrature3D Q, Basis3D B>
double integrate_volume(const Interpolation<B>& geometry, const LocalEnergy& energy,
                        Eigen::VectorXd* gradient = nullptr, Eigen::MatrixXd* hessian = nullptr) {
    if (gradient && !energy.has_gradient())
        throw std::logic_error("integrate_volume: energy does not support gradient evaluation");
    if (hessian && !energy.has_hessian())
        throw std::logic_error("integrate_volume: energy does not support Hessian evaluation");
    if (gradient || hessian) {
        const int n = energy.num_dofs();
        if (n <= 0)
            throw std::invalid_argument("integrate_volume requires positive num_dofs");
        if (gradient)
            gradient->setZero(n);
        if (hessian)
            hessian->setZero(n, n);
    }
    double result = 0;
    const auto& points = Q::points();
    const auto& weights = Q::weights();
    for (int q = 0; q < Q::num_points; ++q) {
        const auto point = rest_point(geometry, points.row(q).transpose());
        const double weight = weights[q] * point.jacobian_determinant;
        if (!std::isfinite(weight))
            throw std::invalid_argument("integrate_volume: non-finite volume weight");
        result += weight * energy.value(point);
        if (gradient) {
            const auto local = energy.gradient(point);
            if (local.size() != gradient->size())
                throw std::invalid_argument("integrate_volume: invalid gradient size");
            *gradient += weight * local;
        }
        if (hessian) {
            const auto local = energy.hessian(point);
            if (local.rows() != hessian->rows() || local.cols() != hessian->cols())
                throw std::invalid_argument("integrate_volume: invalid Hessian size");
            *hessian += weight * local;
        }
    }
    return result;
}

} // namespace litepgo
