#pragma once

#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"

#include <stdexcept>
#include <utility>

namespace litepgo {

// Each class binds one affine active field to the material's static derivatives.
template <class Material> class MaterialLocalEnergyF final : public LocalEnergy {
  public:
    explicit MaterialLocalEnergyF(MaterialFields fields) : fields_(std::move(fields)) {
        fields_.validate();
        active_ = std::dynamic_pointer_cast<const AffineField<Eigen::Matrix3d>>(fields_.F);
        if (!active_ || active_->num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergyF requires an affine F field with positive DOFs");
    }

    int num_dofs() const override {
        return active_->num_dofs();
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }

    double value(const RestPointData& point) const override {
        return Material::value(fields_.evaluate(point));
    }
    Eigen::VectorXd gradient(const RestPointData& point) const override {
        const auto J = active_jacobian(point);
        return gradient_wrt_dofs(J, Material::gradient_F(fields_.evaluate(point)));
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        const auto J = active_jacobian(point);
        return hessian_wrt_dofs(J, Material::hessian_F(fields_.evaluate(point)));
    }

  private:
    Eigen::MatrixXd active_jacobian(const RestPointData& point) const {
        auto J = active_->jacobian(point);
        if (J.rows() != 9 || J.cols() != num_dofs() || !J.allFinite())
            throw std::invalid_argument("MaterialLocalEnergyF: invalid active Jacobian");
        return J;
    }

    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::Matrix3d>> active_;
};

template <class Material> class MaterialLocalEnergyFp final : public LocalEnergy {
  public:
    explicit MaterialLocalEnergyFp(MaterialFields fields) : fields_(std::move(fields)) {
        fields_.validate();
        active_ = std::dynamic_pointer_cast<const AffineField<Eigen::Matrix3d>>(fields_.Fp);
        if (!active_ || active_->num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergyFp requires an affine Fp field with positive DOFs");
    }

    int num_dofs() const override {
        return active_->num_dofs();
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }

    double value(const RestPointData& point) const override {
        return Material::value(fields_.evaluate(point));
    }
    Eigen::VectorXd gradient(const RestPointData& point) const override {
        const auto J = active_jacobian(point);
        return gradient_wrt_dofs(J, Material::gradient_Fp(fields_.evaluate(point)));
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        const auto J = active_jacobian(point);
        return hessian_wrt_dofs(J, Material::hessian_Fp(fields_.evaluate(point)));
    }

  private:
    Eigen::MatrixXd active_jacobian(const RestPointData& point) const {
        auto J = active_->jacobian(point);
        if (J.rows() != 9 || J.cols() != num_dofs() || !J.allFinite())
            throw std::invalid_argument("MaterialLocalEnergyFp: invalid active Jacobian");
        return J;
    }

    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::Matrix3d>> active_;
};

template <class Material> class MaterialLocalEnergyTheta final : public LocalEnergy {
  public:
    explicit MaterialLocalEnergyTheta(MaterialFields fields) : fields_(std::move(fields)) {
        fields_.validate();
        active_ = std::dynamic_pointer_cast<const AffineField<Eigen::VectorXd>>(fields_.theta);
        if (!active_ || active_->num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergyTheta requires an affine theta field with positive DOFs");
    }

    int num_dofs() const override {
        return active_->num_dofs();
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }

    double value(const RestPointData& point) const override {
        return Material::value(fields_.evaluate(point));
    }
    Eigen::VectorXd gradient(const RestPointData& point) const override {
        const auto J = active_jacobian(point);
        return gradient_wrt_dofs(J, Material::gradient_theta(fields_.evaluate(point)));
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        const auto J = active_jacobian(point);
        return hessian_wrt_dofs(J, Material::hessian_theta(fields_.evaluate(point)));
    }

  private:
    Eigen::MatrixXd active_jacobian(const RestPointData& point) const {
        auto J = active_->jacobian(point);
        if (J.rows() != Material::num_parameters || J.cols() != num_dofs() || !J.allFinite())
            throw std::invalid_argument("MaterialLocalEnergyTheta: invalid active Jacobian");
        return J;
    }

    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::VectorXd>> active_;
};

} // namespace litepgo
