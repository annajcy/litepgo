#pragma once

#include "litepgo/fem/point_energy.hpp"

#include <Eigen/Core>
#include <utility>

namespace litepgo {

// Mapping must be affine in the active local DOFs: its Jacobian depends only
// on the reference point. E differentiates the corresponding point argument.
// Pairing semantics are the caller's responsibility; only types are checked.
template <class MappingType, class QuadratureCache, PointValue E>
    requires std::same_as<typename E::Point, typename MappingType::Point> &&
             requires(const MappingType& mapping, const typename MappingType::Fields& fields,
                      const typename MappingType::ReferencePoint& point, const QuadratureCache& cache) {
                 { mapping.point_input(fields, point) } -> std::same_as<typename MappingType::Point>;
                 { mapping.active_jacobian(point) } -> std::same_as<typename MappingType::Jacobian>;
                 { cache.sample(0).point } -> std::same_as<const typename MappingType::ReferencePoint&>;
                 { cache.sample(0).volume_weight } -> std::convertible_to<double>;
                 { QuadratureCache::num_quadrature_points } -> std::convertible_to<int>;
             }
class LocalKernel final {
  public:
    using Mapping = MappingType;
    using Cache = QuadratureCache;
    using Fields = typename Mapping::Fields;
    static constexpr int num_dofs = Mapping::num_dofs;
    using Gradient = Eigen::Matrix<double, num_dofs, 1>;
    using Hessian = Eigen::Matrix<double, num_dofs, num_dofs>;

    // Store cache, mapping and energy by value; no local field state is retained.
    LocalKernel(Cache cache, Mapping mapping, E energy)
        : cache_(std::move(cache)), mapping_(std::move(mapping)), energy_(std::move(energy)) {}

    double value(const Fields& fields) const {
        double result = 0.0;
        for_each_point(fields, [&](const auto& sample, const auto& input) {
            result += sample.volume_weight * energy_.value(input);
        });
        return result;
    }

    Gradient gradient(const Fields& fields) const
        requires PointFirstOrderEnergy<E> && std::same_as<typename E::Gradient, typename Mapping::PointGradient>
    {
        Gradient result = Gradient::Zero();
        for_each_point(fields, [&](const auto& sample, const auto& input) {
            const auto B = mapping_.active_jacobian(sample.point);
            const typename Mapping::PointGradient g = energy_.gradient(input);
            const Eigen::Map<const Eigen::Matrix<double, Mapping::point_dofs, 1>> vector(g.data());
            result.noalias() += sample.volume_weight * B.transpose() * vector;
        });
        return result;
    }

    // Exact Hessian of the active block, without projection or mixed blocks.
    // Affine mappings do not require the point energy's first derivative here.
    Hessian hessian(const Fields& fields) const
        requires PointSecondOrderEnergy<E> && std::same_as<typename E::Hessian, typename Mapping::PointHessian>
    {
        Hessian result = Hessian::Zero();
        for_each_point(fields, [&](const auto& sample, const auto& input) {
            const auto B = mapping_.active_jacobian(sample.point);
            const typename Mapping::PointHessian H = energy_.hessian(input);
            result.noalias() += sample.volume_weight * B.transpose() * H * B;
        });
        return result;
    }

  private:
    template <class Visitor> void for_each_point(const Fields& fields, Visitor&& visitor) const {
        for (int q = 0; q < Cache::num_quadrature_points; ++q) {
            const auto& sample = cache_.sample(q);
            const auto input = mapping_.point_input(fields, sample.point);
            visitor(sample, input);
        }
    }

    Cache cache_;
    Mapping mapping_;
    E energy_;
};

} // namespace litepgo
