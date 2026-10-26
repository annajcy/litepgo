#pragma once

#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/local/local_energy.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>
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

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyF::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            typename std::decay_t<decltype(Material::gradient_F(inputs))>::PlainObject stress;
            typename std::decay_t<decltype(Material::hessian_F(inputs))>::PlainObject tangent;
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

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyFp::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            typename std::decay_t<decltype(Material::gradient_Fp(inputs))>::PlainObject stress;
            typename std::decay_t<decltype(Material::hessian_Fp(inputs))>::PlainObject tangent;
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

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyTheta::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            typename std::decay_t<decltype(Material::gradient_theta(inputs))>::PlainObject stress;
            typename std::decay_t<decltype(Material::hessian_theta(inputs))>::PlainObject tangent;
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
