#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo::fem {

// Density and derivatives with respect to the F field coefficients. All input fields must outlive the energy.
template <class Material> class VolumetricMaterialLocalEnergyF3D final : public LocalEnergy3D {
  public:
    VolumetricMaterialLocalEnergyF3D(const Field3D& F, const Field3D& Fp, const Field3D& theta)
        : F_(F), Fp_(Fp), theta_(theta) {
        if (F_.num_dofs() <= 0)
            throw std::invalid_argument("VolumetricMaterialLocalEnergyF3D requires positive F DOFs");
    }
    VolumetricMaterialLocalEnergyF3D(const Field3D&&, const Field3D&, const Field3D&) = delete;
    VolumetricMaterialLocalEnergyF3D(const Field3D&, const Field3D&&, const Field3D&) = delete;
    VolumetricMaterialLocalEnergyF3D(const Field3D&, const Field3D&, const Field3D&&) = delete;
    int num_dofs() const override {
        return F_.num_dofs();
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }
    double eval_value(const Eigen::Vector3d& xi) const override {
        const double W = Material::eval_W(eval_inputs(xi));
        if (!std::isfinite(W))
            throw std::domain_error("VolumetricMaterialLocalEnergyF3D: non-finite density");
        return W;
    }
    Eigen::VectorXd eval_gradient(const Eigen::Vector3d& xi) const override {
        const auto input = eval_inputs(xi);
        return eval_gradient_wrt_dofs(eval_F_derivative(xi), Material::eval_dW_dF(input));
    }
    Eigen::MatrixXd eval_hessian(const Eigen::Vector3d& xi) const override {
        const auto input = eval_inputs(xi);
        return eval_hessian_wrt_dofs(eval_F_derivative(xi), Material::eval_d2W_dF2(input));
    }
    // Available when Material supplies eval_feasible_step(inputs, delta_inputs).
    // The increment follows the same flattened coefficient order as the derivatives.
    double eval_feasible_step(const Eigen::Vector3d& xi, const Eigen::Ref<const Eigen::VectorXd>& delta_coeffs) const {
        if (delta_coeffs.size() != num_dofs() || !delta_coeffs.allFinite())
            throw std::invalid_argument("VolumetricMaterialLocalEnergyF3D: invalid coefficient increment");
        const auto inputs = eval_inputs(xi);
        const Eigen::VectorXd delta_value = eval_F_derivative(xi) * delta_coeffs;
        if (!delta_value.allFinite())
            throw std::domain_error("VolumetricMaterialLocalEnergyF3D: non-finite input increment");
        VolumetricMaterialInputs delta{Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
                                       Eigen::VectorXd::Zero(inputs.theta.size())};
        delta.F = delta_value.reshaped(3, 3);
        return Material::eval_feasible_step(inputs, delta);
    }
    double accumulate_into(const Eigen::Vector3d& xi, double weight, StridedVecRef* gradient,
                           StridedMatRef* hessian) const override {
        if (!xi.allFinite() || !std::isfinite(weight))
            throw std::invalid_argument("VolumetricMaterialLocalEnergyF3D: invalid parameters or weight");
        const auto input = eval_inputs(xi);
        const double result = weight * Material::eval_W(input);
        if (gradient || hessian) {
            const auto derivative = eval_F_derivative(xi);
            if (gradient)
                add_gradient_wrt_dofs(derivative, Material::eval_dW_dF(input), weight, *gradient, false);
            if (hessian)
                add_hessian_wrt_dofs(derivative, Material::eval_d2W_dF2(input), weight, *hessian, false);
        }
        if (!std::isfinite(result) || (gradient && !gradient->allFinite()))
            throw std::domain_error("VolumetricMaterialLocalEnergyF3D: non-finite accumulation");
        if (hessian && !hessian->allFinite())
            throw std::domain_error("VolumetricMaterialLocalEnergyF3D: non-finite accumulation");
        return result;
    }

  private:
    VolumetricMaterialInputs eval_inputs(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite())
            throw std::invalid_argument("VolumetricMaterialLocalEnergyF3D requires finite parameters");
        const auto F = F_.eval_value(xi);
        const auto Fp = Fp_.eval_value(xi);
        const auto theta = theta_.eval_value(xi);
        if (F.size() != 9 || Fp.size() != 9 || theta.size() == 0 || !F.allFinite() || !Fp.allFinite() ||
            !theta.allFinite())
            throw std::invalid_argument("VolumetricMaterialLocalEnergyF3D requires compatible finite material inputs");
        return {F.reshaped(3, 3), Fp.reshaped(3, 3), theta};
    }
    Eigen::MatrixXd eval_F_derivative(const Eigen::Vector3d& xi) const {
        const auto derivative = F_.eval_dvalue_dcoeffs(xi);
        if (derivative.rows() != 9 || derivative.cols() != num_dofs() || !derivative.allFinite())
            throw std::invalid_argument("VolumetricMaterialLocalEnergyF3D: invalid active coefficient derivative");
        return derivative;
    }
    const Field3D& F_;
    const Field3D& Fp_;
    const Field3D& theta_;
};

} // namespace litepgo::fem
