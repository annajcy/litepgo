#pragma once

#include "litepgo/fem/kinematics/reference_mapping.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/material_inputs.hpp"

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>

namespace litepgo {

enum class MaterialVariable { F, Fp, Theta };

namespace detail {
template <class Value> struct ValueSample {
    const Value value;
};
template <bool NeedCoefficients = true, class Source>
auto material_sample(const Source& source, const Eigen::Vector3d& xi) {
    if constexpr (requires { source.evaluate(xi); } && (NeedCoefficients || !requires { source.value(xi); }))
        return source.evaluate(xi);
    else
        return ValueSample{source.value(xi).eval()};
}
template <class Sample> auto material_value(const Sample& field_sample) {
    if constexpr (requires { field_sample.F(); })
        return field_sample.F();
    else
        return field_sample.value;
}
template <class Sample>
inline constexpr bool has_coefficient_derivative = requires(const Sample& field_sample) {
    field_sample.dvalue_dcoeffs();
} || requires(const Sample& field_sample) { field_sample.dF_du_local(); };
template <class Sample> Eigen::MatrixXd coefficient_derivative(const Sample& field_sample) {
    if constexpr (requires { field_sample.dF_du_local(); })
        return field_sample.dF_du_local();
    else if constexpr (requires { field_sample.dvalue_dcoeffs(); })
        return field_sample.dvalue_dcoeffs();
    else
        throw std::logic_error("Active input requires an affine coefficient derivative");
}
} // namespace detail

// Binds one material input's coefficient block. Other inputs and the reference mapping remain fixed.
template <class Material, MaterialVariable Active, class FSource, class FpSource, class ThetaSource,
          class XField = std::monostate>
class MaterialLocalEnergy final : public LocalEnergy {
    static constexpr int active_index = Active == MaterialVariable::F ? 0 : Active == MaterialVariable::Fp ? 1 : 2;
    using Samples = std::tuple<
        decltype(detail::material_sample < active_index == 0 > (std::declval<const FSource&>(), Eigen::Vector3d{})),
        decltype(detail::material_sample < active_index == 1 > (std::declval<const FpSource&>(), Eigen::Vector3d{})),
        decltype(detail::material_sample < active_index ==
                 2 > (std::declval<const ThetaSource&>(), Eigen::Vector3d{}))>;
    using ActiveSample = std::tuple_element_t<active_index, Samples>;

  public:
    MaterialLocalEnergy(FSource F_source, FpSource Fp_source, ThetaSource theta_source, XField X_field = {})
        : F_source_(std::move(F_source)), Fp_source_(std::move(Fp_source)), theta_source_(std::move(theta_source)),
          X_field_(std::move(X_field)) {
        if constexpr (!detail::has_coefficient_derivative<ActiveSample>)
            throw std::invalid_argument("MaterialLocalEnergy requires an affine active input");
        if (num_dofs() <= 0)
            throw std::invalid_argument("MaterialLocalEnergy requires positive active DOFs");
    }
    int num_dofs() const override {
        const auto& source = [&]() -> const auto& {
            if constexpr (Active == MaterialVariable::F)
                return F_source_;
            else if constexpr (Active == MaterialVariable::Fp)
                return Fp_source_;
            else
                return theta_source_;
        }();
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
        return Material::value(inputs(sample_inputs(xi)));
    }
    Eigen::VectorXd gradient(const Eigen::Vector3d& xi) const override {
        const auto field_samples = sample_inputs(xi);
        return gradient_wrt_dofs(active_derivative(field_samples), material_gradient(inputs(field_samples)));
    }
    Eigen::MatrixXd hessian(const Eigen::Vector3d& xi) const override {
        const auto field_samples = sample_inputs(xi);
        return hessian_wrt_dofs(active_derivative(field_samples), material_hessian(inputs(field_samples)));
    }
    double det_dX_dxi(const Eigen::Vector3d& xi) const override {
        return determinant(detail::material_sample < active_index == 0 > (F_source_, xi), xi);
    }
    double accumulate(const Eigen::Vector3d& xi, double quadrature_weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!xi.allFinite() || !std::isfinite(quadrature_weight))
            throw std::invalid_argument("MaterialLocalEnergy::accumulate requires finite quadrature data");
        // F's sample owns the same mapping used for both constitutive kinematics and the volume weight.
        const auto F_sample = detail::material_sample < active_index == 0 > (F_source_, xi);
        const double dV = quadrature_weight * determinant(F_sample, xi);
        if (!std::isfinite(dV))
            throw std::invalid_argument("MaterialLocalEnergy: non-finite volume weight");
        const Samples field_samples{F_sample, detail::material_sample < active_index == 1 > (Fp_source_, xi),
                                    detail::material_sample < active_index == 2 > (theta_source_, xi)};
        const auto material_inputs = inputs(field_samples);
        const double result = dV * Material::value(material_inputs);
        if (gradient || hessian) {
            const auto derivative = active_derivative(field_samples);
            if (gradient)
                add_gradient_wrt_dofs(derivative, material_gradient(material_inputs), dV, *gradient);
            if (hessian)
                add_hessian_wrt_dofs(derivative, material_hessian(material_inputs), dV, *hessian);
        }
        return result;
    }

  private:
    Samples sample_inputs(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy requires finite parameters");
        return {detail::material_sample < active_index == 0 > (F_source_, xi),
                detail::material_sample < active_index == 1 > (Fp_source_, xi),
                detail::material_sample < active_index == 2 > (theta_source_, xi)};
    }
    static MaterialInputs inputs(const Samples& field_samples) {
        const auto F = detail::material_value(std::get<0>(field_samples));
        const auto Fp = detail::material_value(std::get<1>(field_samples));
        const auto theta = detail::material_value(std::get<2>(field_samples));
        if (F.rows() != 3 || F.cols() != 3 || Fp.rows() != 3 || Fp.cols() != 3 || theta.cols() != 1 ||
            theta.size() == 0 || !F.allFinite() || !Fp.allFinite() || !theta.allFinite())
            throw std::invalid_argument("MaterialLocalEnergy requires compatible finite material inputs");
        return {F, Fp, theta};
    }
    Eigen::MatrixXd active_derivative(const Samples& field_samples) const {
        auto derivative = detail::coefficient_derivative(std::get<active_index>(field_samples));
        Eigen::Index rows = 9;
        if constexpr (Active == MaterialVariable::Theta)
            rows = detail::material_value(std::get<2>(field_samples)).size();
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
    template <class FSample> double determinant(const FSample& F_sample, const Eigen::Vector3d& xi) const {
        const double determinant = [&] {
            if constexpr (requires { F_sample.det_dX_dxi(); })
                return F_sample.det_dX_dxi();
            else
                return ReferenceMapping(X_field_.evaluate(xi)).det_dX_dxi();
        }();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("MaterialLocalEnergy: invalid reference volume");
        return determinant;
    }
    FSource F_source_;
    FpSource Fp_source_;
    ThetaSource theta_source_;
    XField X_field_;
};

// A deformation source already binds its coordinate field; an independent second geometry is not accepted.
template <class Material, MaterialVariable Active, class FSource, class FpSource, class ThetaSource>
auto make_material_local_energy(FSource F_source, FpSource Fp_source, ThetaSource theta_source)
    requires requires(const FSource& source) { source.evaluate(Eigen::Vector3d{}).det_dX_dxi(); }
{
    return MaterialLocalEnergy<Material, Active, FSource, FpSource, ThetaSource>(
        std::move(F_source), std::move(Fp_source), std::move(theta_source));
}
// Direct F coefficients have no coordinate mapping, so bind one explicitly at construction.
template <class Material, MaterialVariable Active, class FSource, class FpSource, class ThetaSource, class XField>
auto make_material_local_energy(FSource F_source, FpSource Fp_source, ThetaSource theta_source, XField X_field)
    requires(!requires(const FSource& source) { source.evaluate(Eigen::Vector3d{}).det_dX_dxi(); })
{
    return MaterialLocalEnergy<Material, Active, FSource, FpSource, ThetaSource, XField>(
        std::move(F_source), std::move(Fp_source), std::move(theta_source), std::move(X_field));
}

} // namespace litepgo
