#pragma once

#include <Eigen/Core>
#include <array>
#include <concepts>
#include <type_traits>

namespace litepgo::fem {

// Coefficient c_i is the reference-coordinate derivative specified here, sampled at point.
template <int Dim> struct CoefficientInfo {
    using Coordinate = std::conditional_t<Dim == 1, double, Eigen::Matrix<double, Dim, 1>>;
    Coordinate point;
    std::array<int, Dim> derivative_order{};
};

// Scalar shape functions; coefficient storage and field components belong to the caller.
// Dim is the reference-coordinate dimension. In 1D xi is a scalar; otherwise it is a vector.
// Values and derivative rows share the basis-function order used by coefficient rows.
// Derivatives are with respect to xi. Second derivatives store the upper triangle once,
// without scaling mixed entries: (11), (11,12,22), or (11,12,13,22,23,33).
// Values and derivatives are evaluated independently. Built-in polynomial bases extrapolate
// outside their reference domains without clamping. Domain conventions are documented per basis.
// coefficient_info describes point/derivative samples in that same row order.
template <class B, int Dim>
concept Basis = (Dim > 0) && requires {
    typename std::integral_constant<int, B::num_basis_functions>;
    requires(B::num_basis_functions > 0);
} && requires(const std::conditional_t<Dim == 1, double, Eigen::Matrix<double, Dim, 1>>& xi, int i) {
    { B::coefficient_info(i) } -> std::same_as<CoefficientInfo<Dim>>;
    { B::eval_N(xi) } -> std::same_as<Eigen::Matrix<double, B::num_basis_functions, 1>>;
    { B::eval_dN_dxi(xi) } -> std::same_as<Eigen::Matrix<double, B::num_basis_functions, Dim>>;
    { B::eval_d2N_dxi2(xi) } -> std::same_as<Eigen::Matrix<double, B::num_basis_functions, Dim * (Dim + 1) / 2>>;
};

template <class B>
concept Basis1D = Basis<B, 1>;

template <class B>
concept Basis2D = Basis<B, 2>;

template <class B>
concept Basis3D = Basis<B, 3>;

} // namespace litepgo::fem
