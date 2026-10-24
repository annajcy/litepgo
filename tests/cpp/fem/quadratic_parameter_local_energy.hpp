#pragma once
#include "litepgo/fem/detail/dof_derivatives.hpp"
#include "litepgo/fem/local_energy.hpp"
#include "litepgo/fem/material_fields.hpp"

// Nonzero parameter Hessian complements NeoHookean's linear theta dependence.
class QuadraticParameterLocalEnergy : public litepgo::LocalValue,
                                      public litepgo::LocalGradient,
                                      public litepgo::LocalHessian {
  public:
    explicit QuadraticParameterLocalEnergy(litepgo::MaterialFields fields)
        : theta_(std::dynamic_pointer_cast<const litepgo::AffineField<Eigen::VectorXd>>(fields.theta)) {
        if (!theta_)
            throw std::invalid_argument("QuadraticParameterLocalEnergy requires affine parameters");
    }
    int num_dofs() const override {
        return theta_->num_dofs();
    }
    double value(const litepgo::VolumePointData& point) const override {
        return 0.5 * theta_->value(point).squaredNorm();
    }
    Eigen::VectorXd gradient(const litepgo::VolumePointData& point) const override {
        return litepgo::detail::gradient_wrt_dofs(theta_->jacobian(point), theta_->value(point));
    }
    Eigen::MatrixXd hessian(const litepgo::VolumePointData& point) const override {
        const auto J = theta_->jacobian(point);
        return J.transpose() * J;
    }

  private:
    std::shared_ptr<const litepgo::AffineField<Eigen::VectorXd>> theta_;
};
