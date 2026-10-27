#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/material_inputs.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

// Shared density evaluation and coefficient pullbacks. All input fields must outlive the energy.
template <class Material> class MaterialLocalEnergy : public LocalEnergy {
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
    double value(const Eigen::Vector3d& xi) const override {
        return Material::value(inputs(xi));
    }
    Eigen::VectorXd gradient(const Eigen::Vector3d& xi) const override {
        const auto input = inputs(xi);
        return gradient_wrt_dofs(active_derivative(xi), material_gradient(input));
    }
    Eigen::MatrixXd hessian(const Eigen::Vector3d& xi) const override {
        const auto input = inputs(xi);
        return hessian_wrt_dofs(active_derivative(xi), material_hessian(input));
    }
    double accumulate(const Eigen::Vector3d& xi, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!xi.allFinite() || !std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergy::accumulate requires finite parameters and weight");
        const auto input = inputs(xi);
        const double result = weight * Material::value(input);
        if (gradient || hessian) {
            const auto derivative = active_derivative(xi);
            if (gradient)
                add_gradient_wrt_dofs(derivative, material_gradient(input), weight, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(derivative, material_hessian(input), weight, *hessian);
        }
        return result;
    }

  protected:
    MaterialLocalEnergy(const Field& F, const Field& Fp, const Field& theta, const Field& active_field)
        : F_(F), Fp_(Fp), theta_(theta), active_field_(active_field) {
        if (active_field_.num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergy requires positive active DOFs");
    }

  private:
    virtual Eigen::MatrixXd material_gradient(const MaterialInputs& input) const = 0;
    virtual Eigen::MatrixXd material_hessian(const MaterialInputs& input) const = 0;

    MaterialInputs inputs(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy requires finite parameters");
        const auto F = F_.value(xi);
        const auto Fp = Fp_.value(xi);
        const auto theta = theta_.value(xi);
        if (F.size() != 9 || Fp.size() != 9 || theta.size() == 0 || !F.allFinite() || !Fp.allFinite() ||
            !theta.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy requires compatible finite material inputs");
        return {F.reshaped(3, 3), Fp.reshaped(3, 3), theta};
    }
    Eigen::MatrixXd active_derivative(const Eigen::Vector3d& xi) const {
        const auto derivative = active_field_.dvalue_dcoeffs(xi);
        if (derivative.rows() != active_field_.num_components() || derivative.cols() != num_dofs() ||
            !derivative.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy: invalid active coefficient derivative");
        return derivative;
    }
    const Field& F_;
    const Field& Fp_;
    const Field& theta_;
    const Field& active_field_;
};

template <class Material> class MaterialLocalEnergyF final : public MaterialLocalEnergy<Material> {
    using Base = MaterialLocalEnergy<Material>;

  public:
    MaterialLocalEnergyF(const Field& F, const Field& Fp, const Field& theta) : Base(F, Fp, theta, F) {}
    MaterialLocalEnergyF(const Field&&, const Field&, const Field&) = delete;
    MaterialLocalEnergyF(const Field&, const Field&&, const Field&) = delete;
    MaterialLocalEnergyF(const Field&, const Field&, const Field&&) = delete;

  private:
    Eigen::MatrixXd material_gradient(const MaterialInputs& input) const override {
        return Material::dW_dF(input);
    }
    Eigen::MatrixXd material_hessian(const MaterialInputs& input) const override {
        return Material::d2W_dF2(input);
    }
};

template <class Material> class MaterialLocalEnergyFp final : public MaterialLocalEnergy<Material> {
    using Base = MaterialLocalEnergy<Material>;

  public:
    MaterialLocalEnergyFp(const Field& F, const Field& Fp, const Field& theta) : Base(F, Fp, theta, Fp) {}
    MaterialLocalEnergyFp(const Field&&, const Field&, const Field&) = delete;
    MaterialLocalEnergyFp(const Field&, const Field&&, const Field&) = delete;
    MaterialLocalEnergyFp(const Field&, const Field&, const Field&&) = delete;

  private:
    Eigen::MatrixXd material_gradient(const MaterialInputs& input) const override {
        return Material::dW_dFp(input);
    }
    Eigen::MatrixXd material_hessian(const MaterialInputs& input) const override {
        return Material::d2W_dFp2(input);
    }
};

template <class Material> class MaterialLocalEnergyTheta final : public MaterialLocalEnergy<Material> {
    using Base = MaterialLocalEnergy<Material>;

  public:
    MaterialLocalEnergyTheta(const Field& F, const Field& Fp, const Field& theta) : Base(F, Fp, theta, theta) {}
    MaterialLocalEnergyTheta(const Field&&, const Field&, const Field&) = delete;
    MaterialLocalEnergyTheta(const Field&, const Field&&, const Field&) = delete;
    MaterialLocalEnergyTheta(const Field&, const Field&, const Field&&) = delete;

  private:
    Eigen::MatrixXd material_gradient(const MaterialInputs& input) const override {
        return Material::dW_dtheta(input);
    }
    Eigen::MatrixXd material_hessian(const MaterialInputs& input) const override {
        return Material::d2W_dtheta2(input);
    }
};

} // namespace litepgo
