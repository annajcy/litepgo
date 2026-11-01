#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem {

// Owns fixed reference geometry and integrates a caller-supplied local energy.
// Rebuild after changing rest geometry. Const evaluations share no scratch storage.
template <Quadrature3D Q> class VolumetricElement3D final {
  public:
    explicit VolumetricElement3D(const Field3D& X_field) {
        if (X_field.num_components() != 3 || !Q::points().allFinite() || !Q::weights().allFinite())
            throw std::invalid_argument("VolumetricElement3D: invalid reference components or quadrature");
        for (int q = 0; q < Q::num_points; ++q) {
            const Eigen::Vector3d xi = Q::points().row(q).transpose();
            const auto derivative = X_field.eval_dvalue_dxi(xi);
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

    // Resizes and overwrites independently nullable outputs in local-energy DOF order.
    // Outputs must not alias local-energy inputs. The local energy owns all density
    // and derivative calculations; accumulate receives the complete volume weight.
    double eval(const LocalEnergy3D& local_energy, Eigen::VectorXd* gradient = nullptr,
                Eigen::MatrixXd* hessian = nullptr) const {
        const int n = local_energy.num_dofs();
        if (n <= 0)
            throw std::invalid_argument("VolumetricElement3D: local energy requires positive DOFs");
        if ((gradient && !local_energy.has_gradient()) || (hessian && !local_energy.has_hessian()))
            throw std::logic_error("VolumetricElement3D: unsupported local-energy derivative");
        if (gradient)
            gradient->setZero(n);
        if (hessian)
            hessian->setZero(n, n);
        double E = 0;
        for (const auto& point : points_) {
            E += local_energy.accumulate_for_integration(point.xi, point.w_dV, gradient, hessian);
        }
        if (!std::isfinite(E))
            throw std::domain_error("VolumetricElement3D: non-finite integrated energy");
        if ((gradient && gradient->size() != n) || (hessian && (hessian->rows() != n || hessian->cols() != n)))
            throw std::logic_error("VolumetricElement3D: invalid local-energy derivative dimensions");
        if ((gradient && !gradient->allFinite()) || (hessian && !hessian->allFinite()))
            throw std::domain_error("VolumetricElement3D: non-finite integrated derivatives");
        return E;
    }

    // Overwrites pre-sized views, allowing direct integration into batch output blocks.
    double eval_into(const LocalEnergy3D& local_energy, StridedVecRef* gradient, StridedMatRef* hessian) const {
        const int n = local_energy.num_dofs();
        if (n <= 0 || (gradient && gradient->size() != n) ||
            (hessian && (hessian->rows() != n || hessian->cols() != n)))
            throw std::invalid_argument("VolumetricElement3D::eval_into: invalid output dimensions");
        if ((gradient && !local_energy.has_gradient()) || (hessian && !local_energy.has_hessian()))
            throw std::logic_error("VolumetricElement3D: unsupported local-energy derivative");
        if (gradient)
            gradient->setZero();
        if (hessian)
            hessian->setZero();
        double E = 0;
        for (const auto& point : points_)
            E += local_energy.accumulate_into(point.xi, point.w_dV, gradient, hessian);
        if (!std::isfinite(E) || (gradient && !gradient->allFinite()) || (hessian && !hessian->allFinite()))
            throw std::domain_error("VolumetricElement3D: non-finite integrated outputs");
        return E;
    }

    // Same cached geometry and complete weights as the energy overload.
    // Resizes and overwrites outputs; the Jacobian is num_outputs x num_inputs.
    Eigen::VectorXd eval(const LocalResponse3D& local_response, Eigen::MatrixXd* jacobian = nullptr) const {
        const int m = local_response.num_outputs(), n = local_response.num_inputs();
        if (m <= 0 || n <= 0)
            throw std::invalid_argument("VolumetricElement3D: local response requires positive dimensions");
        if (jacobian && !local_response.has_jacobian())
            throw std::logic_error("VolumetricElement3D: unsupported local-response Jacobian");
        Eigen::VectorXd value = Eigen::VectorXd::Zero(m);
        if (jacobian)
            jacobian->setZero(m, n);
        for (const auto& point : points_)
            local_response.accumulate_for_integration(point.xi, point.w_dV, value, jacobian);
        if (value.size() != m || (jacobian && (jacobian->rows() != m || jacobian->cols() != n)))
            throw std::logic_error("VolumetricElement3D: invalid local-response dimensions");
        if (!value.allFinite() || (jacobian && !jacobian->allFinite()))
            throw std::domain_error("VolumetricElement3D: non-finite integrated response");
        return value;
    }

    // Response counterpart of the energy view overload; no output allocation.
    void eval_into(const LocalResponse3D& local_response, StridedVecRef value,
                   StridedMatRef* jacobian = nullptr) const {
        const int m = local_response.num_outputs(), n = local_response.num_inputs();
        if (m <= 0 || n <= 0 || value.size() != m || (jacobian && (jacobian->rows() != m || jacobian->cols() != n)))
            throw std::invalid_argument("VolumetricElement3D::eval_into: invalid response dimensions");
        if (jacobian && !local_response.has_jacobian())
            throw std::logic_error("VolumetricElement3D: unsupported local-response Jacobian");
        value.setZero();
        if (jacobian)
            jacobian->setZero();
        for (const auto& point : points_)
            local_response.accumulate_into(point.xi, point.w_dV, value, jacobian);
        if (!value.allFinite() || (jacobian && !jacobian->allFinite()))
            throw std::domain_error("VolumetricElement3D: non-finite integrated response");
    }

    // LocalEnergy supplies eval_feasible_step(xi, delta_coeffs) when this is used.
    // shortcut: samples quadrature points; add spatial certification when an entire
    // curved/high-order element must remain feasible between quadrature points.
    template <class LocalEnergy>
    double max_feasible_step(const LocalEnergy& local_energy,
                             const Eigen::Ref<const Eigen::VectorXd>& delta_coeffs) const {
        if (delta_coeffs.size() != local_energy.num_dofs() || delta_coeffs.size() == 0 || !delta_coeffs.allFinite())
            throw std::invalid_argument("VolumetricElement3D: invalid local coefficient increment");
        double alpha = 1;
        for (const auto& point : points_) {
            const double step = local_energy.eval_feasible_step(point.xi, delta_coeffs);
            if (!(step >= 0 && step <= 1))
                throw std::logic_error("VolumetricElement3D: local feasible step outside [0,1]");
            alpha = std::min(alpha, step);
        }
        return alpha;
    }

  private:
    struct Point {
        Eigen::Vector3d xi;
        double w_dV;
    };
    std::array<Point, Q::num_points> points_;
};

} // namespace litepgo::fem
