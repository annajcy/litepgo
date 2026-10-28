#pragma once

#include "litepgo/fem/basis/basis_sample.hpp"

#include <concepts>
#include <type_traits>

namespace litepgo {

// Dim is the parameter-domain dimension, not the number of basis functions or
// field components. In 1D the parameter is a scalar; otherwise it is a vector.
// Samples provide values and first and packed second reference derivatives.
template <class B, int Dim>
concept Basis = (Dim > 0) &&
                requires {
                    typename std::integral_constant<int, B::num_basis_functions>;
                    requires(B::num_basis_functions > 0);
                } &&
                requires(const std::conditional_t<Dim == 1, double, Eigen::Matrix<double, Dim, 1>>& xi,
                         const typename B::Sample& basis_sample) {
                    { B::evaluate(xi) } -> std::same_as<typename B::Sample>;
                    { (basis_sample.N) } -> std::same_as<const Eigen::Matrix<double, B::num_basis_functions, 1>&>;
                    {
                        (basis_sample.dN_dxi)
                    } -> std::same_as<const Eigen::Matrix<double, B::num_basis_functions, Dim>&>;
                    {
                        (basis_sample.d2N_dxi2)
                    } -> std::same_as<const Eigen::Matrix<double, B::num_basis_functions, Dim * (Dim + 1) / 2>&>;
                };

template <class B>
concept Basis1D = Basis<B, 1>;

template <class B>
concept Basis2D = Basis<B, 2>;

template <class B>
concept Basis3D = Basis<B, 3>;

} // namespace litepgo
