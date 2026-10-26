#pragma once

#include <Eigen/Core>

#include <concepts>
#include <type_traits>

namespace litepgo {

// Dim is the parameter-domain dimension, not the number of basis functions or
// field components. In 1D the parameter is a scalar; otherwise it is a vector.
// Sample stores N[a] = N_a(xi) and dN_dxi(a,d) = dN_a/dxi_d.
template <class B, int Dim>
concept Basis = (Dim > 0) &&
                requires {
                    typename std::integral_constant<int, B::num_basis_functions>;
                    requires(B::num_basis_functions > 0);
                } &&
                requires(const std::conditional_t<Dim == 1, double, Eigen::Matrix<double, Dim, 1>>& xi,
                         const typename B::Sample& sample) {
                    { B::evaluate(xi) } -> std::same_as<typename B::Sample>;
                    { (sample.N) } -> std::same_as<const Eigen::Matrix<double, B::num_basis_functions, 1>&>;
                    { (sample.dN_dxi) } -> std::same_as<const Eigen::Matrix<double, B::num_basis_functions, Dim>&>;
                };

template <class B>
concept Basis1D = Basis<B, 1>;

template <class B>
concept Basis3D = Basis<B, 3>;

} // namespace litepgo
