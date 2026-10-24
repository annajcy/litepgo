#pragma once

#include "litepgo/fem/local_energy.hpp"

#include <type_traits>
#include <utility>

namespace litepgo {

// Integrates local energies using cached geometry points and weights.
// The local energy handles field evaluation and chain rules.
template <class QuadratureCache>
    requires requires(const QuadratureCache& cache) {
        cache.sample(0).point;
        { cache.sample(0).volume_weight } -> std::convertible_to<double>;
        { QuadratureCache::num_quadrature_points } -> std::convertible_to<int>;
    }
class LocalKernel final {
    using GeometryPoint = std::remove_cvref_t<decltype(std::declval<const QuadratureCache&>().sample(0).point)>;

  public:
    explicit LocalKernel(QuadratureCache cache) : cache_(std::move(cache)) {}

    template <LocalValue<GeometryPoint> Energy> double value(const Energy& energy) const {
        double result = 0;
        for_each_point([&](const auto& sample) { result += sample.volume_weight * energy.value(sample.point); });
        return result;
    }

    template <LocalFirstOrderEnergy<GeometryPoint> Energy>
    Eigen::Matrix<double, Energy::num_dofs, 1> gradient(const Energy& energy) const {
        Eigen::Matrix<double, Energy::num_dofs, 1> result = Eigen::Matrix<double, Energy::num_dofs, 1>::Zero();
        for_each_point([&](const auto& sample) { result += sample.volume_weight * energy.gradient(sample.point); });
        return result;
    }

    template <LocalSecondOrderEnergy<GeometryPoint> Energy>
    Eigen::Matrix<double, Energy::num_dofs, Energy::num_dofs> hessian(const Energy& energy) const {
        Eigen::Matrix<double, Energy::num_dofs, Energy::num_dofs> result =
            Eigen::Matrix<double, Energy::num_dofs, Energy::num_dofs>::Zero();
        for_each_point([&](const auto& sample) { result += sample.volume_weight * energy.hessian(sample.point); });
        return result;
    }

  private:
    template <class Visitor> void for_each_point(Visitor&& visitor) const {
        for (int q = 0; q < QuadratureCache::num_quadrature_points; ++q)
            visitor(cache_.sample(q));
    }

    QuadratureCache cache_;
};

} // namespace litepgo
