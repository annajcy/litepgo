#pragma once

#include "litepgo/fem/material/material_inputs.hpp"

namespace litepgo {

// W = det(Fp) [mu/2 (||Fe||^2 - 3) - mu log(Je) + lambda/2 log(Je)^2],
// Fe = F Fp^-1, Je = det(Fe). Density is per original reference volume.
// Requires finite theta = [lambda, mu], lambda >= 0, mu > 0,
// and finite F/Fp (invalid_argument otherwise). Non-positive det(Fp)/det(Fe)
// or unrepresentable kinematics throw InfeasibleState.
// Derivatives hold the other arguments fixed. Hessians use column-major vec.
class NeoHookean final {
  public:
    static constexpr int num_parameters = 2;

    double value(const MaterialInputs& inputs) const;
    Eigen::Matrix3d gradient_F(const MaterialInputs& inputs) const;
    Eigen::Matrix<double, 9, 9> hessian_F(const MaterialInputs& inputs) const;
    Eigen::Matrix3d gradient_Fp(const MaterialInputs& inputs) const;
    Eigen::Matrix<double, 9, 9> hessian_Fp(const MaterialInputs& inputs) const;
    Eigen::Vector2d gradient_theta(const MaterialInputs& inputs) const;
    Eigen::Matrix2d hessian_theta(const MaterialInputs& inputs) const;
};

} // namespace litepgo
