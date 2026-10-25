#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/geometry/interpolation.hpp"

#include <cmath>

namespace litepgo {

// Owns displacement coefficients; returns I + grad_X u (F or Fp).
template <Basis3D B> class DeformationGradientField final : public AffineField<Eigen::Matrix3d> {
  public:
    explicit DeformationGradientField(Eigen::MatrixXd displacement) : interpolation_(std::move(displacement)) {
        if (interpolation_.num_components() != 3)
            throw std::invalid_argument("DeformationGradientField requires three displacement components");
    }
    int num_dofs() const override {
        return interpolation_.num_dofs();
    }
    Eigen::Matrix3d value(const RestPointData& point) const override {
        check_point(point);
        Eigen::Matrix3d result =
            Eigen::Matrix3d::Identity() + interpolation_.gradient(point.xi) * point.inverse_jacobian;
        if (!result.allFinite())
            throw std::invalid_argument("DeformationGradientField::value: non-finite result");
        return result;
    }
    Eigen::MatrixXd jacobian(const RestPointData& point) const override {
        check_point(point);
        Eigen::MatrixXd result =
            Eigen::kroneckerProduct(point.inverse_jacobian.transpose(), Eigen::Matrix3d::Identity()) *
            interpolation_.gradient_matrix(point.xi);
        if (!result.allFinite())
            throw std::invalid_argument("DeformationGradientField::jacobian: non-finite result");
        return result;
    }

  private:
    static void check_point(const RestPointData& point) {
        if (!std::isfinite(point.jacobian_determinant) || point.jacobian_determinant <= 0.0 ||
            !point.inverse_jacobian.allFinite())
            throw std::invalid_argument("DeformationGradientField requires valid volume geometry");
    }
    Interpolation<B> interpolation_;
};

} // namespace litepgo
