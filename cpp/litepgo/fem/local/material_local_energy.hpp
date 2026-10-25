#pragma once

#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/local/local_energy.hpp"

#include <cmath>
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
        Eigen::VectorXd result = Eigen::VectorXd::Zero(num_dofs());
        active_->accumulate_derivatives(point, Material::gradient_F(fields_.evaluate(point)), Eigen::MatrixXd{}, 1,
                                        &result, nullptr);
        return result;
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(num_dofs(), num_dofs());
        active_->accumulate_derivatives(point, Eigen::MatrixXd{}, Material::hessian_F(fields_.evaluate(point)), 1,
                                        nullptr, &result);
        return result;
    }

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyF::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            Eigen::Matrix3d stress;
            Eigen::Matrix<double, 9, 9> tangent;
            if (gradient)
                stress = Material::gradient_F(inputs);
            if (hessian)
                tangent = Material::hessian_F(inputs);
            active_->accumulate_derivatives(point, stress, tangent, weight, gradient, hessian);
        }
        return result;
    }

  private:
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
        Eigen::VectorXd result = Eigen::VectorXd::Zero(num_dofs());
        active_->accumulate_derivatives(point, Material::gradient_Fp(fields_.evaluate(point)), Eigen::MatrixXd{}, 1,
                                        &result, nullptr);
        return result;
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(num_dofs(), num_dofs());
        active_->accumulate_derivatives(point, Eigen::MatrixXd{}, Material::hessian_Fp(fields_.evaluate(point)), 1,
                                        nullptr, &result);
        return result;
    }

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyFp::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            Eigen::Matrix3d stress;
            Eigen::Matrix<double, 9, 9> tangent;
            if (gradient)
                stress = Material::gradient_Fp(inputs);
            if (hessian)
                tangent = Material::hessian_Fp(inputs);
            active_->accumulate_derivatives(point, stress, tangent, weight, gradient, hessian);
        }
        return result;
    }

  private:
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
        Eigen::VectorXd result = Eigen::VectorXd::Zero(num_dofs());
        active_->accumulate_derivatives(point, Material::gradient_theta(fields_.evaluate(point)), Eigen::MatrixXd{}, 1,
                                        &result, nullptr);
        return result;
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(num_dofs(), num_dofs());
        active_->accumulate_derivatives(point, Eigen::MatrixXd{}, Material::hessian_theta(fields_.evaluate(point)), 1,
                                        nullptr, &result);
        return result;
    }

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyTheta::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            Eigen::VectorXd stress;
            Eigen::MatrixXd tangent;
            if (gradient)
                stress = Material::gradient_theta(inputs);
            if (hessian)
                tangent = Material::hessian_theta(inputs);
            active_->accumulate_derivatives(point, stress, tangent, weight, gradient, hessian);
        }
        return result;
    }

  private:
    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::VectorXd>> active_;
};

} // namespace litepgo
