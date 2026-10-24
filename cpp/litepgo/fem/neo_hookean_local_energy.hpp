#pragma once

#include "litepgo/fem/detail/dof_derivatives.hpp"
#include "litepgo/fem/neo_hookean.hpp"

#include <type_traits>
#include <utility>

namespace litepgo {

// Each object owns its field snapshot and binds one constitutive derivative
// to the corresponding affine field. Values are densities, without quadrature weights.

template <class Fields>
    requires(Fields::num_parameters == NeoHookean::num_parameters) && requires(const Fields& fields) {
        fields.F;
        requires(std::remove_cvref_t<decltype(fields.F)>::num_dofs > 0);
    }
class NeoHookeanLocalEnergyF final {
  public:
    static constexpr int num_dofs = std::remove_cvref_t<decltype(std::declval<Fields>().F)>::num_dofs;

    explicit NeoHookeanLocalEnergyF(Fields fields) : fields_(std::move(fields)) {}

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            { fields.evaluate(point) } -> std::same_as<NeoHookean::Point>;
        }
    double value(const GeometryPoint& point) const {
        return NeoHookean{}.value(fields_.evaluate(point));
    }

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            {
                detail::gradient_wrt_dofs(fields.F.jacobian(point),
                                          NeoHookeanEnergyF{}.gradient(fields.evaluate(point)))
            } -> std::same_as<Eigen::Matrix<double, num_dofs, 1>>;
        }
    Eigen::Matrix<double, num_dofs, 1> gradient(const GeometryPoint& point) const {
        return detail::gradient_wrt_dofs(fields_.F.jacobian(point),
                                         NeoHookeanEnergyF{}.gradient(fields_.evaluate(point)));
    }

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            {
                detail::hessian_wrt_dofs(fields.F.jacobian(point), NeoHookeanEnergyF{}.hessian(fields.evaluate(point)))
            } -> std::same_as<Eigen::Matrix<double, num_dofs, num_dofs>>;
        }
    Eigen::Matrix<double, num_dofs, num_dofs> hessian(const GeometryPoint& point) const {
        return detail::hessian_wrt_dofs(fields_.F.jacobian(point),
                                        NeoHookeanEnergyF{}.hessian(fields_.evaluate(point)));
    }

  private:
    Fields fields_;
};

template <class Fields>
    requires(Fields::num_parameters == NeoHookean::num_parameters) && requires(const Fields& fields) {
        fields.Fp;
        requires(std::remove_cvref_t<decltype(fields.Fp)>::num_dofs > 0);
    }
class NeoHookeanLocalEnergyFp final {
  public:
    static constexpr int num_dofs = std::remove_cvref_t<decltype(std::declval<Fields>().Fp)>::num_dofs;

    explicit NeoHookeanLocalEnergyFp(Fields fields) : fields_(std::move(fields)) {}

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            { fields.evaluate(point) } -> std::same_as<NeoHookean::Point>;
        }
    double value(const GeometryPoint& point) const {
        return NeoHookean{}.value(fields_.evaluate(point));
    }

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            {
                detail::gradient_wrt_dofs(fields.Fp.jacobian(point),
                                          NeoHookeanEnergyFp{}.gradient(fields.evaluate(point)))
            } -> std::same_as<Eigen::Matrix<double, num_dofs, 1>>;
        }
    Eigen::Matrix<double, num_dofs, 1> gradient(const GeometryPoint& point) const {
        return detail::gradient_wrt_dofs(fields_.Fp.jacobian(point),
                                         NeoHookeanEnergyFp{}.gradient(fields_.evaluate(point)));
    }

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            {
                detail::hessian_wrt_dofs(fields.Fp.jacobian(point),
                                         NeoHookeanEnergyFp{}.hessian(fields.evaluate(point)))
            } -> std::same_as<Eigen::Matrix<double, num_dofs, num_dofs>>;
        }
    Eigen::Matrix<double, num_dofs, num_dofs> hessian(const GeometryPoint& point) const {
        return detail::hessian_wrt_dofs(fields_.Fp.jacobian(point),
                                        NeoHookeanEnergyFp{}.hessian(fields_.evaluate(point)));
    }

  private:
    Fields fields_;
};

template <class Fields>
    requires(Fields::num_parameters == NeoHookean::num_parameters) && requires(const Fields& fields) {
        fields.theta;
        requires(std::remove_cvref_t<decltype(fields.theta)>::num_dofs > 0);
    }
class NeoHookeanLocalEnergyTheta final {
  public:
    static constexpr int num_dofs = std::remove_cvref_t<decltype(std::declval<Fields>().theta)>::num_dofs;

    explicit NeoHookeanLocalEnergyTheta(Fields fields) : fields_(std::move(fields)) {}

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            { fields.evaluate(point) } -> std::same_as<NeoHookean::Point>;
        }
    double value(const GeometryPoint& point) const {
        return NeoHookean{}.value(fields_.evaluate(point));
    }

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            {
                detail::gradient_wrt_dofs(fields.theta.jacobian(point),
                                          NeoHookeanEnergyTheta{}.gradient(fields.evaluate(point)))
            } -> std::same_as<Eigen::Matrix<double, num_dofs, 1>>;
        }
    Eigen::Matrix<double, num_dofs, 1> gradient(const GeometryPoint& point) const {
        return detail::gradient_wrt_dofs(fields_.theta.jacobian(point),
                                         NeoHookeanEnergyTheta{}.gradient(fields_.evaluate(point)));
    }

    template <class GeometryPoint>
        requires requires(const Fields& fields, const GeometryPoint& point) {
            {
                detail::hessian_wrt_dofs(fields.theta.jacobian(point),
                                         NeoHookeanEnergyTheta{}.hessian(fields.evaluate(point)))
            } -> std::same_as<Eigen::Matrix<double, num_dofs, num_dofs>>;
        }
    Eigen::Matrix<double, num_dofs, num_dofs> hessian(const GeometryPoint& point) const {
        return detail::hessian_wrt_dofs(fields_.theta.jacobian(point),
                                        NeoHookeanEnergyTheta{}.hessian(fields_.evaluate(point)));
    }

  private:
    Fields fields_;
};

} // namespace litepgo
