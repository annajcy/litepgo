#pragma once

#include "litepgo/fem/material_types.hpp"

#include <concepts>
#include <type_traits>

namespace litepgo {

// W is energy per reference volume. Derivatives hold the other inputs fixed.
// Matrix gradients retain input shape; Hessians use vec(A)[i + 3*j] = A(i,j).
// Hessians are exact, without PSD projection. These concepts check signatures,
// not mathematical correctness or thread safety.
template <class M>
concept VolumetricMaterial = requires {
    typename std::integral_constant<int, M::num_parameters>;
    requires(M::num_parameters >= 0);
    requires std::same_as<typename M::Parameters, typename MaterialTypes<M::num_parameters>::Parameters>;
} && requires(const M& material, const Mat3& F, const Mat3& Fp, const typename M::Parameters& theta) {
    { material.value(F, Fp, theta) } -> std::same_as<double>;
    { material.gradient_F(F, Fp, theta) } -> std::same_as<Mat3>;
    { material.hessian_F(F, Fp, theta) } -> std::same_as<Mat9>;
};

// Each optional capability is independent, including first and second derivatives.
template <class M>
concept MaterialWithFpGradient = VolumetricMaterial<M> && requires(const M& material, const Mat3& F, const Mat3& Fp,
                                                                   const typename M::Parameters& theta) {
    { material.gradient_Fp(F, Fp, theta) } -> std::same_as<Mat3>;
};

template <class M>
concept MaterialWithFpHessian = VolumetricMaterial<M> && requires(const M& material, const Mat3& F, const Mat3& Fp,
                                                                  const typename M::Parameters& theta) {
    { material.hessian_Fp(F, Fp, theta) } -> std::same_as<Mat9>;
};

template <class M>
concept MaterialWithThetaGradient = VolumetricMaterial<M> && requires(const M& material, const Mat3& F, const Mat3& Fp,
                                                                      const typename M::Parameters& theta) {
    { material.gradient_theta(F, Fp, theta) } -> std::same_as<typename M::Parameters>;
};

template <class M>
concept MaterialWithThetaHessian = VolumetricMaterial<M> && requires(const M& material, const Mat3& F, const Mat3& Fp,
                                                                     const typename M::Parameters& theta) {
    {
        material.hessian_theta(F, Fp, theta)
    } -> std::same_as<typename MaterialTypes<M::num_parameters>::ParameterHessian>;
};

} // namespace litepgo
