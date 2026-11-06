#pragma once

#include "litepgo/energy/response/response.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"
#include <Eigen/LU>
#include <unsupported/Eigen/KroneckerProduct>

#include <array>
#include <stdexcept>

namespace litepgo::fem {

// Fixed geometry and basis make vec(F_q) = vec(I) + B_q * coefficients.
// Owns B_q in row-major displacement-coefficient order; rebuild if geometry changes.
template <Basis3D B, Quadrature3D Q> class DeformationGradientKinematics3D final {
  public:
    explicit DeformationGradientKinematics3D(energy::Response& X) {
        if (X.num_inputs() != 3 || X.num_outputs() != 3)
            throw std::invalid_argument("DeformationGradientKinematics3D: geometry must map R3 to R3");
        for (int q = 0; q < Q::num_points; ++q) {
            const Eigen::Vector3d xi = Q::points().row(q).transpose();
            const Eigen::Matrix3d J = X.eval_jacobian(xi);
            const double determinant = J.determinant();
            if (!std::isfinite(determinant) || determinant <= 0)
                throw std::invalid_argument("DeformationGradientKinematics3D: positive geometry determinant required");
            const Eigen::MatrixXd gradient = B::eval_dN_dxi(xi) * J.inverse();
            derivatives_[q] = Eigen::kroneckerProduct(gradient.transpose(), Eigen::Matrix3d::Identity());
            if (!derivatives_[q].allFinite())
                throw std::invalid_argument("DeformationGradientKinematics3D: non-finite physical gradient");
        }
    }

    const Eigen::MatrixXd& derivative(int q) const {
        return derivatives_.at(static_cast<std::size_t>(q));
    }

    Eigen::Matrix3d eval(int q, const Eigen::Ref<const Eigen::VectorXd>& coefficients) const {
        if (coefficients.size() != 3 * B::num_basis_functions || !coefficients.allFinite())
            throw std::invalid_argument("DeformationGradientKinematics3D: invalid displacement coefficients");
        const Eigen::VectorXd increment = derivative(q) * coefficients;
        const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + increment.reshaped(3, 3);
        if (!F.allFinite())
            throw std::domain_error("DeformationGradientKinematics3D: non-finite deformation gradient");
        return F;
    }

  private:
    std::array<Eigen::MatrixXd, Q::num_points> derivatives_;
};

} // namespace litepgo::fem
