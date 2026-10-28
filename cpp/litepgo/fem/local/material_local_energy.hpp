#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/material_inputs.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

// Shared density evaluation and coefficient pullbacks. All input fields must outlive the energy.
template <class Material> class MaterialLocalEnergy3D : public LocalEnergy3D {
  public:
    int num_dofs() const override {
        return active_field_.num_dofs();
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }
    double eval_value(const Eigen::Vector3d& xi) const override {
        return Material::eval_W(eval_inputs(xi));
    }
    Eigen::VectorXd eval_gradient(const Eigen::Vector3d& xi) const override {
        const auto input = eval_inputs(xi);
        return eval_gradient_wrt_dofs(eval_active_derivative(xi), eval_material_gradient(input));
    }
    Eigen::MatrixXd eval_hessian(const Eigen::Vector3d& xi) const override {
        const auto input = eval_inputs(xi);
        return eval_hessian_wrt_dofs(eval_active_derivative(xi), eval_material_hessian(input));
    }
    double accumulate(const Eigen::Vector3d& xi, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!xi.allFinite() || !std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergy3D::accumulate requires finite parameters and weight");
        const auto input = eval_inputs(xi);
        const double result = weight * Material::eval_W(input);
        if (gradient || hessian) {
            const auto derivative = eval_active_derivative(xi);
            if (gradient)
                add_gradient_wrt_dofs(derivative, eval_material_gradient(input), weight, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(derivative, eval_material_hessian(input), weight, *hessian);
        }
        return result;
    }

  protected:
    MaterialLocalEnergy3D(const Field3D& F, const Field3D& Fp, const Field3D& theta, const Field3D& active_field)
        : F_(F), Fp_(Fp), theta_(theta), active_field_(active_field) {
        if (active_field_.num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergy3D requires positive active DOFs");
    }

  private:
    virtual Eigen::MatrixXd eval_material_gradient(const MaterialInputs& input) const = 0;
    virtual Eigen::MatrixXd eval_material_hessian(const MaterialInputs& input) const = 0;

    MaterialInputs eval_inputs(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy3D requires finite parameters");
        const auto F = F_.eval_value(xi);
        const auto Fp = Fp_.eval_value(xi);
        const auto theta = theta_.eval_value(xi);
        if (F.size() != 9 || Fp.size() != 9 || theta.size() == 0 || !F.allFinite() || !Fp.allFinite() ||
            !theta.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy3D requires compatible finite material inputs");
        return {F.reshaped(3, 3), Fp.reshaped(3, 3), theta};
    }
    Eigen::MatrixXd eval_active_derivative(const Eigen::Vector3d& xi) const {
        const auto derivative = active_field_.eval_dvalue_dcoeffs(xi);
        if (derivative.rows() != active_field_.num_components() || derivative.cols() != num_dofs() ||
            !derivative.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy3D: invalid active coefficient derivative");
        return derivative;
    }
    const Field3D& F_;
    const Field3D& Fp_;
    const Field3D& theta_;
    const Field3D& active_field_;
};

template <class Material> class MaterialLocalEnergyF3D final : public MaterialLocalEnergy3D<Material> {
    using Base = MaterialLocalEnergy3D<Material>;

  public:
    MaterialLocalEnergyF3D(const Field3D& F, const Field3D& Fp, const Field3D& theta) : Base(F, Fp, theta, F) {}
    MaterialLocalEnergyF3D(const Field3D&&, const Field3D&, const Field3D&) = delete;
    MaterialLocalEnergyF3D(const Field3D&, const Field3D&&, const Field3D&) = delete;
    MaterialLocalEnergyF3D(const Field3D&, const Field3D&, const Field3D&&) = delete;

  private:
    Eigen::MatrixXd eval_material_gradient(const MaterialInputs& input) const override {
        return Material::eval_dW_dF(input);
    }
    Eigen::MatrixXd eval_material_hessian(const MaterialInputs& input) const override {
        return Material::eval_d2W_dF2(input);
    }
};

template <class Material> class MaterialLocalEnergyFp3D final : public MaterialLocalEnergy3D<Material> {
    using Base = MaterialLocalEnergy3D<Material>;

  public:
    MaterialLocalEnergyFp3D(const Field3D& F, const Field3D& Fp, const Field3D& theta) : Base(F, Fp, theta, Fp) {}
    MaterialLocalEnergyFp3D(const Field3D&&, const Field3D&, const Field3D&) = delete;
    MaterialLocalEnergyFp3D(const Field3D&, const Field3D&&, const Field3D&) = delete;
    MaterialLocalEnergyFp3D(const Field3D&, const Field3D&, const Field3D&&) = delete;

  private:
    Eigen::MatrixXd eval_material_gradient(const MaterialInputs& input) const override {
        return Material::eval_dW_dFp(input);
    }
    Eigen::MatrixXd eval_material_hessian(const MaterialInputs& input) const override {
        return Material::eval_d2W_dFp2(input);
    }
};

template <class Material> class MaterialLocalEnergyTheta3D final : public MaterialLocalEnergy3D<Material> {
    using Base = MaterialLocalEnergy3D<Material>;

  public:
    MaterialLocalEnergyTheta3D(const Field3D& F, const Field3D& Fp, const Field3D& theta) : Base(F, Fp, theta, theta) {}
    MaterialLocalEnergyTheta3D(const Field3D&&, const Field3D&, const Field3D&) = delete;
    MaterialLocalEnergyTheta3D(const Field3D&, const Field3D&&, const Field3D&) = delete;
    MaterialLocalEnergyTheta3D(const Field3D&, const Field3D&, const Field3D&&) = delete;

  private:
    Eigen::MatrixXd eval_material_gradient(const MaterialInputs& input) const override {
        return Material::eval_dW_dtheta(input);
    }
    Eigen::MatrixXd eval_material_hessian(const MaterialInputs& input) const override {
        return Material::eval_d2W_dtheta2(input);
    }
};

} // namespace litepgo
