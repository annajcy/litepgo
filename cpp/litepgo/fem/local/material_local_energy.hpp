#pragma once

#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/material_inputs.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace litepgo {

enum class MaterialVariable { F, Fp, Theta };

// Binds one material input's coefficient block. Other inputs and the coordinate field remain fixed.
template <class Material, MaterialVariable Active, class FSource, class FpSource, class ThetaSource,
          class XField = std::monostate>
class MaterialLocalEnergy final : public LocalEnergy {
    using ActiveSource = std::conditional_t<Active == MaterialVariable::F, FSource,
                                            std::conditional_t<Active == MaterialVariable::Fp, FpSource, ThetaSource>>;

  public:
    MaterialLocalEnergy(FSource F_source, FpSource Fp_source, ThetaSource theta_source, XField X_field = {})
        : F_source_(std::move(F_source)), Fp_source_(std::move(Fp_source)), theta_source_(std::move(theta_source)),
          X_field_(std::move(X_field)) {
        if constexpr (!requires(const ActiveSource& source) { source.dvalue_dcoeffs(Eigen::Vector3d{}); })
            throw std::invalid_argument("MaterialLocalEnergy requires an affine active input");
        if (num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergy requires positive active DOFs");
    }
    int num_dofs() const override {
        const auto& source = active_source();
        if constexpr (requires { source.num_dofs(); })
            return source.num_dofs();
        else
            return 0;
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
        return gradient_wrt_dofs(active_derivative(input, xi), material_gradient(input));
    }
    Eigen::MatrixXd hessian(const Eigen::Vector3d& xi) const override {
        const auto input = inputs(xi);
        return hessian_wrt_dofs(active_derivative(input, xi), material_hessian(input));
    }
    double det_dX_dxi(const Eigen::Vector3d& xi) const override {
        const auto& coordinates = coordinate_field();
        if (!xi.allFinite() || coordinates.num_components() != 3)
            throw std::invalid_argument("MaterialLocalEnergy requires finite parameters and three coordinates");
        const auto derivative = coordinates.dvalue_dxi(xi);
        if (derivative.rows() != 3 || derivative.cols() != 3 || !derivative.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy requires a finite 3-by-3 coordinate derivative");
        const double determinant = Eigen::Matrix3d(derivative).determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("MaterialLocalEnergy requires a finite positive coordinate determinant");
        return determinant;
    }
    double accumulate(const Eigen::Vector3d& xi, double quadrature_weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!xi.allFinite() || !std::isfinite(quadrature_weight))
            throw std::invalid_argument("MaterialLocalEnergy::accumulate requires finite quadrature data");
        const double dV = quadrature_weight * det_dX_dxi(xi);
        if (!std::isfinite(dV))
            throw std::invalid_argument("MaterialLocalEnergy: non-finite volume weight");
        const auto input = inputs(xi);
        const double result = dV * Material::value(input);
        if (gradient || hessian) {
            const auto derivative = active_derivative(input, xi);
            if (gradient)
                add_gradient_wrt_dofs(derivative, material_gradient(input), dV, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(derivative, material_hessian(input), dV, *hessian);
        }
        return result;
    }

  private:
    const ActiveSource& active_source() const {
        if constexpr (Active == MaterialVariable::F)
            return F_source_;
        else if constexpr (Active == MaterialVariable::Fp)
            return Fp_source_;
        else
            return theta_source_;
    }
    MaterialInputs inputs(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy requires finite parameters");
        const auto F = F_source_.value(xi);
        const auto Fp = Fp_source_.value(xi);
        const auto theta = theta_source_.value(xi);
        if (F.size() != 9 || Fp.size() != 9 || theta.cols() != 1 || theta.size() == 0 || !F.allFinite() ||
            !Fp.allFinite() || !theta.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy requires compatible finite material inputs");
        return {F.reshaped(3, 3), Fp.reshaped(3, 3), theta};
    }
    Eigen::MatrixXd active_derivative(const MaterialInputs& input, const Eigen::Vector3d& xi) const {
        const auto& source = active_source();
        Eigen::MatrixXd derivative;
        if constexpr (requires { source.dvalue_dcoeffs(xi); })
            derivative = source.dvalue_dcoeffs(xi);
        else
            throw std::logic_error("Active input requires an affine coefficient derivative");
        const Eigen::Index rows = Active == MaterialVariable::Theta ? input.theta.size() : 9;
        if (derivative.rows() != rows || derivative.cols() != num_dofs() || !derivative.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy: invalid active coefficient derivative");
        return derivative;
    }
    static auto material_gradient(const MaterialInputs& input) {
        if constexpr (Active == MaterialVariable::F)
            return Material::dW_dF(input);
        else if constexpr (Active == MaterialVariable::Fp)
            return Material::dW_dFp(input);
        else
            return Material::dW_dtheta(input);
    }
    static auto material_hessian(const MaterialInputs& input) {
        if constexpr (Active == MaterialVariable::F)
            return Material::d2W_dF2(input);
        else if constexpr (Active == MaterialVariable::Fp)
            return Material::d2W_dFp2(input);
        else
            return Material::d2W_dtheta2(input);
    }
    const auto& coordinate_field() const {
        if constexpr (requires { F_source_.coordinate_field(); })
            return F_source_.coordinate_field();
        else
            return X_field_;
    }
    FSource F_source_;
    FpSource Fp_source_;
    ThetaSource theta_source_;
    XField X_field_;
};

// A deformation field already binds its coordinate field; a second geometry is not accepted.
template <class Material, MaterialVariable Active, class FSource, class FpSource, class ThetaSource>
auto make_material_local_energy(FSource F_source, FpSource Fp_source, ThetaSource theta_source)
    requires requires(const FSource& source) { source.coordinate_field(); }
{
    return MaterialLocalEnergy<Material, Active, FSource, FpSource, ThetaSource>(
        std::move(F_source), std::move(Fp_source), std::move(theta_source));
}
// Direct F coefficients have no coordinate mapping, so bind one explicitly at construction.
template <class Material, MaterialVariable Active, class FSource, class FpSource, class ThetaSource, class XField>
auto make_material_local_energy(FSource F_source, FpSource Fp_source, ThetaSource theta_source, XField X_field)
    requires(!requires(const FSource& source) { source.coordinate_field(); })
{
    return MaterialLocalEnergy<Material, Active, FSource, FpSource, ThetaSource, XField>(
        std::move(F_source), std::move(Fp_source), std::move(theta_source), std::move(X_field));
}

} // namespace litepgo
