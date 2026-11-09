#pragma once
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include <Eigen/LU>
#include <array>
#include <memory>

namespace litepgo::fem {
// Spatial query vec(I + Du(xi) DX(xi)^-1), with column-major matrix packing.
// Children remain fixed. Its spatial Jacobian uses their Hessians; second derivatives
// would require third spatial derivatives and are not supported.
class DeformationGradientResponse final : public energy::FixedPatternResponse {
  public:
    DeformationGradientResponse(std::shared_ptr<energy::Response> displacement,
                                std::shared_ptr<energy::Response> geometry)
        : FixedPatternResponse(Eigen::MatrixXd::Ones(9, 3).sparseView()), displacement_(std::move(displacement)),
          geometry_(std::move(geometry)) {
        for (const auto& child : {displacement_, geometry_})
            if (!child || child->num_inputs() != 3 || child->num_outputs() != 3 || !child->has_hessian())
                throw std::invalid_argument(
                    "DeformationGradientResponse: children must map R3 to R3 and support Hessians");
    }

  private:
    Vec eval_jvp_impl(const ConstVecRef& xi, const ConstVecRef& v) override {
        return product(xi, v, false);
    }
    Vec eval_vjp_impl(const ConstVecRef& xi, const ConstVecRef& v) override {
        return product(xi, v, true);
    }
    Vec product(const ConstVecRef& xi, const ConstVecRef& v, bool transpose) {
        const Eigen::Matrix3d U = displacement_->eval_jacobian(xi);
        const Eigen::Matrix3d X = geometry_->eval_jacobian(xi);
        const double determinant = X.determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::domain_error("DeformationGradientResponse: positive geometry determinant required");
        const Eigen::Matrix3d inverse = X.inverse(), gradient = U * inverse;
        Eigen::Matrix3d dU, dX;
        Eigen::Vector3d result = Eigen::Vector3d::Zero();
        Eigen::Matrix3d adjoint, geometry_adjoint;
        if (transpose) {
            adjoint = v.reshaped(3, 3) * inverse.transpose();
            geometry_adjoint = -gradient.transpose() * adjoint;
        }
        for (int component = 0; component < 3; ++component) {
            const Vec w = Vec::Unit(3, component);
            const Eigen::Matrix3d Hu = displacement_->eval_hessian(xi, w);
            const Eigen::Matrix3d Hx = geometry_->eval_hessian(xi, w);
            if (transpose) {
                result += Hu * adjoint.row(component).transpose() + Hx * geometry_adjoint.row(component).transpose();
            } else {
                dU.row(component) = (Hu * v).transpose();
                dX.row(component) = (Hx * v).transpose();
            }
        }
        if (transpose)
            return result;
        const Eigen::Matrix3d derivative = (dU - gradient * dX) * inverse;
        return derivative.reshaped();
    }
    Vec eval_impl(const ConstVecRef& xi, SpMat* J) override {
        const Eigen::Matrix3d U = displacement_->eval_jacobian(xi);
        const Eigen::Matrix3d X = geometry_->eval_jacobian(xi);
        const double determinant = X.determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::domain_error("DeformationGradientResponse: positive geometry determinant required");
        const Eigen::Matrix3d inverse = X.inverse();
        const Eigen::Matrix3d gradient = U * inverse;
        if (J) {
            std::array<Eigen::Matrix3d, 3> dU, dX;
            for (int component = 0; component < 3; ++component) {
                const Vec w = Vec::Unit(3, component);
                const Eigen::Matrix3d Hu = displacement_->eval_hessian(xi, w);
                const Eigen::Matrix3d Hx = geometry_->eval_hessian(xi, w);
                for (int k = 0; k < 3; ++k) {
                    dU[k].row(component) = Hu.col(k).transpose();
                    dX[k].row(component) = Hx.col(k).transpose();
                }
            }
            for (int k = 0; k < 3; ++k) {
                const Eigen::Matrix3d derivative = (dU[k] - gradient * dX[k]) * inverse;
                const Vec flattened = derivative.reshaped();
                for (SpMat::InnerIterator it(*J, k); it; ++it)
                    it.valueRef() = flattened[it.row()];
            }
        }
        const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + gradient;
        return F.reshaped();
    }
    std::shared_ptr<energy::Response> displacement_, geometry_;
};
} // namespace litepgo::fem
