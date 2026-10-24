#pragma once

#include "litepgo/fem/point_energy.hpp"

#include <Eigen/Core>
#include <type_traits>
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
        return integrate<0>(fields);
    }

    Gradient gradient(const Fields& fields) const
        requires PointFirstOrderEnergy<E> && std::same_as<typename E::Gradient, typename Mapping::PointGradient>
    {
        return integrate<1>(fields);
    }

    // Exact Hessian of the active block, without projection or mixed blocks.
    // Affine mappings do not require the point energy's first derivative here.
    Hessian hessian(const Fields& fields) const
        requires PointSecondOrderEnergy<E> && std::same_as<typename E::Hessian, typename Mapping::PointHessian>
    {
        return integrate<2>(fields);
    }

  private:
    template <int Order> auto integrate(const Fields& fields) const {
        using Result = std::conditional_t<Order == 0, double, std::conditional_t<Order == 1, Gradient, Hessian>>;
        Result result;
        if constexpr (Order == 0)
            result = 0.0;
        else
            result.setZero();
        for (int q = 0; q < Cache::num_quadrature_points; ++q) {
            const auto& sample = cache_.sample(q);
            const auto input = mapping_.point_input(fields, sample.point);
            const double weight = sample.volume_weight;
            if constexpr (Order == 0) {
                result += weight * energy_.value(input);
            } else {
                const auto B = mapping_.active_jacobian(sample.point);
                if constexpr (Order == 1) {
                    const typename Mapping::PointGradient g = energy_.gradient(input);
                    const Eigen::Map<const Eigen::Matrix<double, Mapping::point_dofs, 1>> vector(g.data());
                    result.noalias() += weight * B.transpose() * vector;
                } else {
                    result.noalias() += weight * B.transpose() * energy_.hessian(input) * B;
                }
            }
        }
        return result;
    }

    Cache cache_;
    Mapping mapping_;
    E energy_;
};

} // namespace litepgo
