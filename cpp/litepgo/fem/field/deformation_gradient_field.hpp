#pragma once

#include "litepgo/fem/field/field.hpp"

#include <Eigen/LU>
#include <cmath>
#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo {

// Borrows u and X; both must outlive this field. Only u's coefficients are active.
class DeformationGradientField3D final : public Field3D {
  public:
    DeformationGradientField3D(const Field3D& u_field, const Field3D& X_field) : u_field_(u_field), X_field_(X_field) {
        if (u_field_.num_components() != 3 || X_field_.num_components() != 3)
            throw std::invalid_argument(
                "DeformationGradientField3D requires three displacement and coordinate components");
    }
    DeformationGradientField3D(const Field3D&&, const Field3D&) = delete;
    DeformationGradientField3D(const Field3D&, const Field3D&&) = delete;

    int num_components() const override {
        return 9;
    }
    int num_dofs() const override {
        return u_field_.num_dofs();
    }
    const Field3D& coordinate_field() const {
        return X_field_;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d& xi) const override {
        const Eigen::Matrix3d dxi_dX = eval_dxi_dX(xi);
        const auto du_dxi = u_field_.eval_dvalue_dxi(xi);
        if (u_field_.num_components() != 3 || du_dxi.rows() != 3 || du_dxi.cols() != 3 || !du_dxi.allFinite())
            throw std::invalid_argument("DeformationGradientField3D requires finite three-component displacement data");
        const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + du_dxi * dxi_dX;
        if (!F.allFinite())
            throw std::invalid_argument("DeformationGradientField3D::eval_value: non-finite result");
        return F.reshaped();
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d& xi) const override {
        const Eigen::Matrix3d dxi_dX = eval_dxi_dX(xi);
        const Eigen::MatrixXd dgrad_xi_dcoeffs = u_field_.eval_dgrad_xi_dcoeffs(xi);
        if (dgrad_xi_dcoeffs.rows() != 9 || dgrad_xi_dcoeffs.cols() != num_dofs() || num_dofs() <= 0 ||
            !dgrad_xi_dcoeffs.allFinite())
            throw std::invalid_argument("DeformationGradientField3D::eval_dvalue_dcoeffs: invalid mixed derivative");
        Eigen::MatrixXd derivative =
            Eigen::kroneckerProduct(dxi_dX.transpose(), Eigen::Matrix3d::Identity()) * dgrad_xi_dcoeffs;
        if (!derivative.allFinite())
            throw std::invalid_argument("DeformationGradientField3D::eval_dvalue_dcoeffs: non-finite result");
        return derivative;
    }

  private:
    Eigen::Matrix3d eval_dxi_dX(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite() || X_field_.num_components() != 3)
            throw std::invalid_argument("DeformationGradientField3D requires finite parameters and three coordinates");
        const auto derivative = X_field_.eval_dvalue_dxi(xi);
        if (derivative.rows() != 3 || derivative.cols() != 3 || !derivative.allFinite())
            throw std::invalid_argument("DeformationGradientField3D requires a finite 3-by-3 coordinate derivative");
        const Eigen::Matrix3d dX_dxi = derivative;
        const double determinant = dX_dxi.determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("DeformationGradientField3D requires a finite positive coordinate determinant");
        const Eigen::Matrix3d inverse = dX_dxi.inverse();
        if (!inverse.allFinite())
            throw std::invalid_argument("DeformationGradientField3D: non-finite inverse coordinate derivative");
        return inverse;
    }
    const Field3D& u_field_;
    const Field3D& X_field_;
};

} // namespace litepgo
