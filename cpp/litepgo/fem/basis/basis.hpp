#pragma once

#include <Eigen/Core>
#include <concepts>
#include <type_traits>

namespace litepgo {

// Dim is the parameter-domain dimension, not the number of basis functions or
// field components. In 1D the parameter is a scalar; otherwise it is a vector.
// Values and reference derivatives are evaluated independently. Second derivatives
// store the upper triangle once, without scaling: (11), (11,12,22), or (11,12,13,22,23,33).
template <class B, int Dim>
concept Basis = (Dim > 0) && requires {
    typename std::integral_constant<int, B::num_basis_functions>;
    requires(B::num_basis_functions > 0);
} && requires(const std::conditional_t<Dim == 1, double, Eigen::Matrix<double, Dim, 1>>& xi) {
    { B::evaluate_N(xi) } -> std::same_as<Eigen::Matrix<double, B::num_basis_functions, 1>>;
    { B::evaluate_dN_dxi(xi) } -> std::same_as<Eigen::Matrix<double, B::num_basis_functions, Dim>>;
    { B::evaluate_d2N_dxi2(xi) } -> std::same_as<Eigen::Matrix<double, B::num_basis_functions, Dim * (Dim + 1) / 2>>;
};

template <class B>
concept Basis1D = Basis<B, 1>;

template <class B>
concept Basis2D = Basis<B, 2>;

template <class B>
concept Basis3D = Basis<B, 3>;

} // namespace litepgo
