#pragma once

#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

#include <array>
#include <stdexcept>

namespace litepgo::fem {

// Fixed geometry and basis make vec(F_q) = vec(I) + B_q * coefficients.
// Owns B_q in row-major displacement-coefficient order; rebuild if geometry changes.
template <Basis3D B, Quadrature3D Q> class DeformationGradientKinematics3D final {
  public:
    explicit DeformationGradientKinematics3D(const Field3D& X) {
        const InterpolatedField3D<B> zero(Eigen::MatrixXd::Zero(B::num_basis_functions, 3));
        const DeformationGradientField3D F(zero, X);
        for (int q = 0; q < Q::num_points; ++q)
            derivatives_[q] = F.eval_dvalue_dcoeffs(Q::points().row(q).transpose());
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
