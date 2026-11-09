#pragma once

#include "litepgo/energy/response/response.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem {

// Snapshots physical quadrature weights and integrates a caller-supplied energy formula.
// Rebuild after changing rest geometry. Const evaluations share no scratch storage.
template <Quadrature3D Q> class VolumetricElement3D final {
  public:
    struct Point {
        Eigen::Vector3d xi;
        double w_dV;
    };

    explicit VolumetricElement3D(energy::Response& geometry) {
        if ((geometry.num_inputs() != 3 || geometry.num_outputs() != 3) || !Q::points().allFinite() ||
            !Q::weights().allFinite())
            throw std::invalid_argument("VolumetricElement3D: invalid reference components or quadrature");
        for (int q = 0; q < Q::num_points; ++q) {
            const Eigen::Vector3d xi = Q::points().row(q).transpose();
            const Eigen::MatrixXd derivative = geometry.evaluate(xi)->jacobian();
            if (derivative.rows() != 3 || !derivative.allFinite())
                throw std::invalid_argument("VolumetricElement3D: invalid reference Jacobian");
            const Eigen::Matrix3d J_X = derivative;
            const double det_J_X = J_X.determinant();
            const double w_dV = Q::weights()[q] * det_J_X;
            if (!std::isfinite(det_J_X) || det_J_X <= 0 || !std::isfinite(w_dV))
                throw std::invalid_argument("VolumetricElement3D: invalid reference determinant or volume weight");
            points_[static_cast<std::size_t>(q)] = {xi, w_dV};
        }
    }

    const Point& point(int q) const {
        return points_.at(static_cast<std::size_t>(q));
    }

    // accumulate(q, point, g, H) adds weighted derivatives and returns weighted energy.
    // Outputs are zeroed here; callbacks must not rebind the views or alias their inputs.
    template <class Accumulate>
    double eval_into(int num_dofs, Accumulate&& accumulate, StridedVecRef* gradient, StridedMatRef* hessian) const {
        if (num_dofs <= 0 || (gradient && gradient->size() != num_dofs) ||
            (hessian && (hessian->rows() != num_dofs || hessian->cols() != num_dofs)))
            throw std::invalid_argument("VolumetricElement3D::eval_into: invalid output dimensions");
        if (gradient)
            gradient->setZero();
        if (hessian)
            hessian->setZero();
        double E = 0;
        for (int q = 0; q < Q::num_points; ++q)
            E += accumulate(q, points_[q], gradient, hessian);
        if (!std::isfinite(E) || (gradient && !gradient->allFinite()) || (hessian && !hessian->allFinite()))
            throw std::domain_error("VolumetricElement3D: non-finite integrated outputs");
        return E;
    }

    // step(q, point) supplies the feasible bound at one quadrature point.
    // shortcut: samples quadrature points; add spatial certification when an entire
    // curved/high-order element must remain feasible between quadrature points.
    template <class Step> double max_feasible_step(Step&& step) const {
        double alpha = 1;
        for (int q = 0; q < Q::num_points; ++q) {
            const double bound = step(q, points_[q]);
            if (!(bound >= 0 && bound <= 1))
                throw std::logic_error("VolumetricElement3D: local feasible step outside [0,1]");
            alpha = std::min(alpha, bound);
        }
        return alpha;
    }

  private:
    std::array<Point, Q::num_points> points_;
};

} // namespace litepgo::fem
