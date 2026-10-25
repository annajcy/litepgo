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
        const auto gradients = physical_gradients(point);
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(9, num_dofs());
        // d F(c,d) / d u(a,c) = d N_a / d X_d; other component pairs are zero.
        for (int a = 0; a < B::num_basis_functions; ++a)
            for (int d = 0; d < 3; ++d)
                result.block<3, 3>(3 * d, 3 * a).diagonal().setConstant(gradients(a, d));
        return result;
    }

    void accumulate_derivatives(const RestPointData& point, const Eigen::Ref<const Eigen::MatrixXd>& value_gradient,
                                const Eigen::Ref<const Eigen::MatrixXd>& value_hessian, double weight,
                                Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight) || (gradient && (gradient->size() != num_dofs() || value_gradient.size() != 9)) ||
            (hessian && (hessian->rows() != num_dofs() || hessian->cols() != num_dofs() || value_hessian.rows() != 9 ||
                         value_hessian.cols() != 9)))
            throw std::invalid_argument(
                "DeformationGradientField::accumulate_derivatives: invalid dimensions or weight");
        if (!gradient && !hessian)
            return;
        const auto gradients = physical_gradients(point);
        if (gradient) {
            Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> output(gradient->data(), 3, B::num_basis_functions);
            const Eigen::Matrix3d stress = value_gradient.reshaped(3, 3);
            output.noalias() += weight * (stress * gradients.transpose());
        }
        if (hessian) {
            // J's three nonzero entries per column reduce the pullback to 3x3 blocks.
            for (int b = 0; b < B::num_basis_functions; ++b) {
                const Eigen::Matrix<double, 9, 3> right = weight * (gradients(b, 0) * value_hessian.leftCols<3>() +
                                                                    gradients(b, 1) * value_hessian.middleCols<3>(3) +
                                                                    gradients(b, 2) * value_hessian.rightCols<3>());
                // Complete columns let the inner basis loop vectorize.
                for (int c = 0; c < 3; ++c) {
                    double* column = hessian->col(3 * b + c).data();
                    for (int a = 0; a < B::num_basis_functions; ++a)
                        for (int i = 0; i < 3; ++i)
                            column[3 * a + i] += gradients(a, 0) * right(i, c) + gradients(a, 1) * right(i + 3, c) +
                                                 gradients(a, 2) * right(i + 6, c);
                }
            }
        }
    }

  private:
    static Eigen::Matrix<double, B::num_basis_functions, 3> physical_gradients(const RestPointData& point) {
        check_point(point);
        Eigen::Matrix<double, B::num_basis_functions, 3> result =
            B::evaluate(point.xi).gradients * point.inverse_jacobian;
        if (!result.allFinite())
            throw std::invalid_argument("DeformationGradientField requires finite physical basis gradients");
        return result;
    }

    static void check_point(const RestPointData& point) {
        if (!std::isfinite(point.jacobian_determinant) || point.jacobian_determinant <= 0.0 ||
            !point.inverse_jacobian.allFinite() || !point.xi.allFinite())
            throw std::invalid_argument("DeformationGradientField requires valid volume geometry");
    }
    Interpolation<B> interpolation_;
};

} // namespace litepgo
