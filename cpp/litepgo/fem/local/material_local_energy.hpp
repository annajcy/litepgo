#pragma once

#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo {

// Each class binds one affine active field to the material's static derivatives.
// value/gradient/hessian evaluate the pointwise density W and its local DOF derivatives;
// integrate_volume weights and sums these to obtain the element energy E_element.
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
        const auto dF_dx_local = active_dF_dx_local(point);
        return gradient_wrt_dofs(dF_dx_local, Material::dW_dF(fields_.evaluate(point)));
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        const auto dF_dx_local = active_dF_dx_local(point);
        return hessian_wrt_dofs(dF_dx_local, Material::d2W_dF2(fields_.evaluate(point)));
    }

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyF::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            const auto dF_dx_local = active_dF_dx_local(point);
            if (gradient)
                add_gradient_wrt_dofs(dF_dx_local, Material::dW_dF(inputs), weight, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(dF_dx_local, Material::d2W_dF2(inputs), weight, *hessian);
        }
        return result;
    }

  private:
    Eigen::MatrixXd active_dF_dx_local(const RestPointData& point) const {
        auto dF_dx_local = active_->dvalue_dx_local(point);
        if (dF_dx_local.rows() != 9 || dF_dx_local.cols() != num_dofs() || !dF_dx_local.allFinite())
            throw std::invalid_argument("MaterialLocalEnergyF: invalid active Jacobian");
        return dF_dx_local;
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
        const auto dFp_dx_local = active_dFp_dx_local(point);
        return gradient_wrt_dofs(dFp_dx_local, Material::dW_dFp(fields_.evaluate(point)));
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        const auto dFp_dx_local = active_dFp_dx_local(point);
        return hessian_wrt_dofs(dFp_dx_local, Material::d2W_dFp2(fields_.evaluate(point)));
    }

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyFp::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            const auto dFp_dx_local = active_dFp_dx_local(point);
            if (gradient)
                add_gradient_wrt_dofs(dFp_dx_local, Material::dW_dFp(inputs), weight, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(dFp_dx_local, Material::d2W_dFp2(inputs), weight, *hessian);
        }
        return result;
    }

  private:
    Eigen::MatrixXd active_dFp_dx_local(const RestPointData& point) const {
        auto dFp_dx_local = active_->dvalue_dx_local(point);
        if (dFp_dx_local.rows() != 9 || dFp_dx_local.cols() != num_dofs() || !dFp_dx_local.allFinite())
            throw std::invalid_argument("MaterialLocalEnergyFp: invalid active Jacobian");
        return dFp_dx_local;
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
        const auto dtheta_dx_local = active_dtheta_dx_local(point);
        return gradient_wrt_dofs(dtheta_dx_local, Material::dW_dtheta(fields_.evaluate(point)));
    }
    Eigen::MatrixXd hessian(const RestPointData& point) const override {
        const auto dtheta_dx_local = active_dtheta_dx_local(point);
        return hessian_wrt_dofs(dtheta_dx_local, Material::d2W_dtheta2(fields_.evaluate(point)));
    }

    double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergyTheta::accumulate requires a finite weight");
        const auto inputs = fields_.evaluate(point);
        const double result = weight * Material::value(inputs);
        if (gradient || hessian) {
            const auto dtheta_dx_local = active_dtheta_dx_local(point);
            if (gradient)
                add_gradient_wrt_dofs(dtheta_dx_local, Material::dW_dtheta(inputs), weight, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(dtheta_dx_local, Material::d2W_dtheta2(inputs), weight, *hessian);
        }
        return result;
    }

  private:
    Eigen::MatrixXd active_dtheta_dx_local(const RestPointData& point) const {
        auto dtheta_dx_local = active_->dvalue_dx_local(point);
        if (dtheta_dx_local.rows() != Material::num_parameters || dtheta_dx_local.cols() != num_dofs() ||
            !dtheta_dx_local.allFinite())
            throw std::invalid_argument("MaterialLocalEnergyTheta: invalid active Jacobian");
        return dtheta_dx_local;
    }

    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::VectorXd>> active_;
};

} // namespace litepgo
