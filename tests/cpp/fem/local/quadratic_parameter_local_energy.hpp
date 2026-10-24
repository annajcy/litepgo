#pragma once
#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"

// Nonzero parameter Hessian complements NeoHookean's linear theta dependence.
class QuadraticParameterLocalEnergy : public litepgo::LocalEnergy {
  public:
    explicit QuadraticParameterLocalEnergy(litepgo::MaterialFields fields)
        : theta_(std::dynamic_pointer_cast<const litepgo::AffineField<Eigen::VectorXd>>(fields.theta)) {
        if (!theta_)
            throw std::invalid_argument("QuadraticParameterLocalEnergy requires affine parameters");
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }
    int num_dofs() const override {
        return theta_->num_dofs();
    }
    double value(const litepgo::VolumePointData& point) const override {
        return 0.5 * theta_->value(point).squaredNorm();
    }
    Eigen::VectorXd gradient(const litepgo::VolumePointData& point) const override {
        return litepgo::gradient_wrt_dofs(theta_->jacobian(point), theta_->value(point));
    }
    Eigen::MatrixXd hessian(const litepgo::VolumePointData& point) const override {
        const auto J = theta_->jacobian(point);
        return J.transpose() * J;
    }

  private:
    std::shared_ptr<const litepgo::AffineField<Eigen::VectorXd>> theta_;
};
