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
        const auto point = rest_point(geometry, points.row(q).transpose());
        const double dV = weights[q] * point.det_dX_dxi;
        if (!std::isfinite(dV))
            throw std::invalid_argument("integrate_volume: non-finite volume weight");
        if (gradient || hessian) {
            E_element += energy.accumulate(point, dV, gradient, hessian);
            if ((gradient && gradient->size() != n) || (hessian && (hessian->rows() != n || hessian->cols() != n)))
                throw std::invalid_argument("integrate_volume: accumulator changed output dimensions");
        } else
            E_element += dV * energy.value(point);
    }
    return E_element;
}

} // namespace litepgo
