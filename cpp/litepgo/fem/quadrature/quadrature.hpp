#pragma once

#include <Eigen/Core>

#include <concepts>
#include <type_traits>

namespace litepgo::fem {

// Points are rows in parameter coordinates; weights form a column vector.
// The concept checks the interface, not the domain or integration accuracy.
template <class Q, int Dim>
concept Quadrature = (Dim > 0) && requires {
    typename std::integral_constant<int, Q::num_points>;
    requires(Q::num_points > 0);
} && requires {
    { Q::points() } -> std::same_as<const Eigen::Matrix<double, Q::num_points, Dim>&>;
    { Q::weights() } -> std::same_as<const Eigen::Matrix<double, Q::num_points, 1>&>;
};

template <class Q>
concept Quadrature1D = Quadrature<Q, 1>;

template <class Q>
concept Quadrature2D = Quadrature<Q, 2>;

template <class Q>
concept Quadrature3D = Quadrature<Q, 3>;

} // namespace litepgo::fem
