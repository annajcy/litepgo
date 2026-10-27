#pragma once

#include "litepgo/fem/field/field.hpp"

#include <Eigen/LU>
#include <cmath>
#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo {

// Borrows u and X; both must outlive this field. Only u's coefficients are active.
class DeformationGradientField final : public Field {
  public:
    DeformationGradientField(const Field& u_field, const Field& X_field) : u_field_(u_field), X_field_(X_field) {
        if (u_field_.num_components() != 3 || X_field_.num_components() != 3)
            throw std::invalid_argument(
                "DeformationGradientField requires three displacement and coordinate components");
    }
    DeformationGradientField(const Field&&, const Field&) = delete;
    DeformationGradientField(const Field&, const Field&&) = delete;

    int num_components() const override {
        return 9;
    }
    int num_dofs() const override {
        return u_field_.num_dofs();
    }
    const Field& coordinate_field() const {
        return X_field_;
    }
    Eigen::VectorXd value(const Eigen::Vector3d& xi) const override {
        const Eigen::Matrix3d dxi_dX = inverse_coordinate_derivative(xi);
        const auto du_dxi = u_field_.dvalue_dxi(xi);
        if (u_field_.num_components() != 3 || du_dxi.rows() != 3 || du_dxi.cols() != 3 || !du_dxi.allFinite())
            throw std::invalid_argument("DeformationGradientField requires finite three-component displacement data");
        const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + du_dxi * dxi_dX;
        if (!F.allFinite())
            throw std::invalid_argument("DeformationGradientField::value: non-finite result");
        return F.reshaped();
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d& xi) const override {
        const Eigen::Matrix3d dxi_dX = inverse_coordinate_derivative(xi);
        const Eigen::MatrixXd dgrad_xi_dcoeffs = u_field_.dgrad_xi_dcoeffs(xi);
        if (dgrad_xi_dcoeffs.rows() != 9 || dgrad_xi_dcoeffs.cols() != num_dofs() || num_dofs() <= 0 ||
            !dgrad_xi_dcoeffs.allFinite())
            throw std::invalid_argument("DeformationGradientField::dvalue_dcoeffs: invalid mixed derivative");
        Eigen::MatrixXd derivative =
            Eigen::kroneckerProduct(dxi_dX.transpose(), Eigen::Matrix3d::Identity()) * dgrad_xi_dcoeffs;
        if (!derivative.allFinite())
            throw std::invalid_argument("DeformationGradientField::dvalue_dcoeffs: non-finite result");
        return derivative;
    }

  private:
    Eigen::Matrix3d inverse_coordinate_derivative(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite() || X_field_.num_components() != 3)
            throw std::invalid_argument("DeformationGradientField requires finite parameters and three coordinates");
        const auto derivative = X_field_.dvalue_dxi(xi);
        if (derivative.rows() != 3 || derivative.cols() != 3 || !derivative.allFinite())
            throw std::invalid_argument("DeformationGradientField requires a finite 3-by-3 coordinate derivative");
        const Eigen::Matrix3d dX_dxi = derivative;
        const double determinant = dX_dxi.determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("DeformationGradientField requires a finite positive coordinate determinant");
        const Eigen::Matrix3d inverse = dX_dxi.inverse();
        if (!inverse.allFinite())
            throw std::invalid_argument("DeformationGradientField: non-finite inverse coordinate derivative");
        return inverse;
    }
    const Field& u_field_;
    const Field& X_field_;
};

} // namespace litepgo
