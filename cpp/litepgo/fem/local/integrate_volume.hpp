#pragma once

#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

// Geometry and quadrature must use the same parameter domain.
// Integrates the density and requested DOF derivatives in one pass. Outputs are
// independently nullable, resized and overwritten; they must not alias energy data.
// Unsupported derivative requests throw std::logic_error before any outputs change.
template <Quadrature3D Q>
double integrate_volume(const LocalEnergy& energy, Eigen::VectorXd* gradient = nullptr,
                        Eigen::MatrixXd* hessian = nullptr) {
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
        const Eigen::Vector3d xi = points.row(q).transpose();
        if (!xi.allFinite() || !std::isfinite(weights[q]))
            throw std::invalid_argument("integrate_volume: non-finite quadrature data");
        E_element += energy.accumulate(xi, weights[q], gradient, hessian);
        if ((gradient && gradient->size() != n) || (hessian && (hessian->rows() != n || hessian->cols() != n)))
            throw std::invalid_argument("integrate_volume: accumulator changed output dimensions");
    }
    return E_element;
}

} // namespace litepgo
