#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {

// Geometry, energy fields and quadrature must use the same element and parameter domain.
// Integrates the density and requested DOF derivatives in one pass. Outputs are
// independently nullable, resized and overwritten; they must not alias energy data.
// Unsupported derivative requests throw std::logic_error before any outputs change.
template <Quadrature3D Q>
double integrate_volume(const Field& X_field, const LocalEnergy& energy, Eigen::VectorXd* gradient = nullptr,
                        Eigen::MatrixXd* hessian = nullptr) {
    if (gradient && !energy.has_gradient())
        throw std::logic_error("integrate_volume: energy does not support gradient evaluation");
    if (hessian && !energy.has_hessian())
        throw std::logic_error("integrate_volume: energy does not support Hessian evaluation");
    if (X_field.num_components() != 3)
        throw std::invalid_argument("integrate_volume requires three coordinate components");
    const int n = gradient || hessian ? energy.num_dofs() : 0;
    if (gradient || hessian) {
        if (n <= 0)
            throw std::invalid_argument("integrate_volume requires positive num_dofs");
        if (gradient)
            gradient->setZero(n);
        if (hessian)
            hessian->setZero(n, n);
    }
    double E_element = 0;
    const auto& points = Q::points();
    const auto& weights = Q::weights();
    for (int q = 0; q < Q::num_points; ++q) {
        const Eigen::Vector3d xi = points.row(q).transpose();
        if (!xi.allFinite() || !std::isfinite(weights[q]))
            throw std::invalid_argument("integrate_volume: non-finite quadrature data");
        const auto derivative = X_field.dvalue_dxi(xi);
        if (derivative.rows() != 3 || derivative.cols() != 3 || !derivative.allFinite())
            throw std::invalid_argument("integrate_volume requires a finite 3-by-3 coordinate derivative");
        const double determinant = Eigen::Matrix3d(derivative).determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("integrate_volume requires a finite positive coordinate determinant");
        const double weight = weights[q] * determinant;
        if (!std::isfinite(weight))
            throw std::invalid_argument("integrate_volume: non-finite volume weight");
        E_element += energy.accumulate(xi, weight, gradient, hessian);
        if ((gradient && gradient->size() != n) || (hessian && (hessian->rows() != n || hessian->cols() != n)))
            throw std::invalid_argument("integrate_volume: accumulator changed output dimensions");
    }
    return E_element;
}

} // namespace litepgo
