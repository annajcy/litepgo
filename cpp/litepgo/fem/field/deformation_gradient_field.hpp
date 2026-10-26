#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/interpolation/interpolation.hpp"

#include <cmath>

namespace litepgo {

// Owns displacement coefficients u_coeffs; returns I + grad_X u (F or Fp).
template <Basis3D B> class DeformationGradientField final : public AffineField<Eigen::Matrix3d> {
  public:
    explicit DeformationGradientField(Eigen::MatrixXd u_coeffs) : interpolation_(std::move(u_coeffs)) {
        if (interpolation_.num_components() != 3)
            throw std::invalid_argument("DeformationGradientField requires three displacement components");
    }
    int num_dofs() const override {
        return interpolation_.num_dofs();
    }
    Eigen::Matrix3d value(const RestPointData& point) const override {
        check_point(point);
        Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + interpolation_.dvalue_dxi(point.xi) * point.dxi_dX;
        if (!F.allFinite())
            throw std::invalid_argument("DeformationGradientField::value: non-finite result");
        return F;
    }
    Eigen::MatrixXd dvalue_dx_local(const RestPointData& point) const override {
        check_point(point);
        const Eigen::Matrix<double, B::num_basis_functions, 3> dN_dX = B::evaluate(point.xi).dN_dxi * point.dxi_dX;
        if (!dN_dX.allFinite())
            throw std::invalid_argument("DeformationGradientField::dvalue_dx_local: non-finite result");
        Eigen::MatrixXd dF_dx_local = Eigen::MatrixXd::Zero(9, num_dofs());
        // d F(c,d) / d u(a,c) = d N_a / d X_d; other component pairs are zero.
        for (int a = 0; a < B::num_basis_functions; ++a)
            for (int d = 0; d < 3; ++d)
                dF_dx_local.block<3, 3>(3 * d, 3 * a).diagonal().setConstant(dN_dX(a, d));
        return dF_dx_local;
    }

  private:
    static void check_point(const RestPointData& point) {
        if (!std::isfinite(point.det_dX_dxi) || point.det_dX_dxi <= 0.0 || !point.dxi_dX.allFinite() ||
            !point.xi.allFinite())
            throw std::invalid_argument("DeformationGradientField requires valid volume geometry");
    }
    Interpolation<B> interpolation_;
};

} // namespace litepgo
