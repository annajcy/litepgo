#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/material_inputs.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

enum class MaterialVariable { F, Fp, Theta };

// Borrows all three fields; they must outlive this energy. Only the selected coefficient block is active.
template <class Material> class MaterialLocalEnergy final : public LocalEnergy {
  public:
    MaterialLocalEnergy(const Field& F, const Field& Fp, const Field& theta, MaterialVariable active)
        : F_(F), Fp_(Fp), theta_(theta), active_(active) {
        if (num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergy requires positive active DOFs");
        if (!has_gradient() || !has_hessian())
            throw std::invalid_argument("MaterialLocalEnergy: material does not support the active derivatives");
    }
    MaterialLocalEnergy(const Field&&, const Field&, const Field&, MaterialVariable) = delete;
    MaterialLocalEnergy(const Field&, const Field&&, const Field&, MaterialVariable) = delete;
    MaterialLocalEnergy(const Field&, const Field&, const Field&&, MaterialVariable) = delete;

    int num_dofs() const override {
        return active_source().num_dofs();
    }
    bool has_gradient() const noexcept override {
        switch (active_) {
        case MaterialVariable::F:
            return requires(const MaterialInputs& input) { Material::dW_dF(input); };
        case MaterialVariable::Fp:
            return requires(const MaterialInputs& input) { Material::dW_dFp(input); };
        case MaterialVariable::Theta:
            return requires(const MaterialInputs& input) { Material::dW_dtheta(input); };
        }
        return false;
    }
    bool has_hessian() const noexcept override {
        switch (active_) {
        case MaterialVariable::F:
            return requires(const MaterialInputs& input) { Material::d2W_dF2(input); };
        case MaterialVariable::Fp:
            return requires(const MaterialInputs& input) { Material::d2W_dFp2(input); };
        case MaterialVariable::Theta:
            return requires(const MaterialInputs& input) { Material::d2W_dtheta2(input); };
        }
        return false;
    }
    double value(const Eigen::Vector3d& xi) const override {
        return Material::value(inputs(xi));
    }
    Eigen::VectorXd gradient(const Eigen::Vector3d& xi) const override {
        const auto input = inputs(xi);
        return gradient_wrt_dofs(active_derivative(input, xi), material_gradient(input));
    }
    Eigen::MatrixXd hessian(const Eigen::Vector3d& xi) const override {
        const auto input = inputs(xi);
        return hessian_wrt_dofs(active_derivative(input, xi), material_hessian(input));
    }
    double accumulate(const Eigen::Vector3d& xi, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!xi.allFinite() || !std::isfinite(weight))
            throw std::invalid_argument("MaterialLocalEnergy::accumulate requires finite parameters and weight");
        const auto input = inputs(xi);
        const double result = weight * Material::value(input);
        if (gradient || hessian) {
            const auto derivative = active_derivative(input, xi);
            if (gradient)
                add_gradient_wrt_dofs(derivative, material_gradient(input), weight, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(derivative, material_hessian(input), weight, *hessian);
        }
        return result;
    }

  private:
    const Field& active_source() const {
        switch (active_) {
        case MaterialVariable::F:
            return F_;
        case MaterialVariable::Fp:
            return Fp_;
        case MaterialVariable::Theta:
            return theta_;
        }
        throw std::invalid_argument("MaterialLocalEnergy: invalid active variable");
    }
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
    Eigen::MatrixXd active_derivative(const MaterialInputs& input, const Eigen::Vector3d& xi) const {
        const auto derivative = active_source().dvalue_dcoeffs(xi);
        const Eigen::Index rows = active_ == MaterialVariable::Theta ? input.theta.size() : 9;
        if (derivative.rows() != rows || derivative.cols() != num_dofs() || !derivative.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy: invalid active coefficient derivative");
        return derivative;
    }
    Eigen::MatrixXd material_gradient(const MaterialInputs& input) const {
        switch (active_) {
        case MaterialVariable::F:
            if constexpr (requires { Material::dW_dF(input); })
                return Material::dW_dF(input);
            break;
        case MaterialVariable::Fp:
            if constexpr (requires { Material::dW_dFp(input); })
                return Material::dW_dFp(input);
            break;
        case MaterialVariable::Theta:
            if constexpr (requires { Material::dW_dtheta(input); })
                return Material::dW_dtheta(input);
            break;
        }
        throw std::logic_error("MaterialLocalEnergy: unsupported material gradient");
    }
    Eigen::MatrixXd material_hessian(const MaterialInputs& input) const {
        switch (active_) {
        case MaterialVariable::F:
            if constexpr (requires { Material::d2W_dF2(input); })
                return Material::d2W_dF2(input);
            break;
        case MaterialVariable::Fp:
            if constexpr (requires { Material::d2W_dFp2(input); })
                return Material::d2W_dFp2(input);
            break;
        case MaterialVariable::Theta:
            if constexpr (requires { Material::d2W_dtheta2(input); })
                return Material::d2W_dtheta2(input);
            break;
        }
        throw std::logic_error("MaterialLocalEnergy: unsupported material Hessian");
    }
    const Field& F_;
    const Field& Fp_;
    const Field& theta_;
    MaterialVariable active_;
};

} // namespace litepgo
