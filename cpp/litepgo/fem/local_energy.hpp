#pragma once

#include <Eigen/Core>
#include <concepts>

namespace litepgo {

template <class Energy, class GeometryPoint>
concept LocalValue = requires(const Energy& energy, const GeometryPoint& point) {
    { energy.value(point) } -> std::same_as<double>;
};

template <class Energy, class GeometryPoint>
concept LocalFirstOrderEnergy =
    LocalValue<Energy, GeometryPoint> && requires(const Energy& energy, const GeometryPoint& point) {
        requires(Energy::num_dofs > 0);
        { energy.gradient(point) } -> std::same_as<Eigen::Matrix<double, Energy::num_dofs, 1>>;
    };

// Deliberately independent of LocalFirstOrderEnergy.
template <class Energy, class GeometryPoint>
concept LocalSecondOrderEnergy =
    LocalValue<Energy, GeometryPoint> && requires(const Energy& energy, const GeometryPoint& point) {
        requires(Energy::num_dofs > 0);
        { energy.hessian(point) } -> std::same_as<Eigen::Matrix<double, Energy::num_dofs, Energy::num_dofs>>;
    };

} // namespace litepgo
