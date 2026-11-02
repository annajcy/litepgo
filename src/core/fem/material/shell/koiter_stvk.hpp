#pragma once

#include "litepgo/fem/material/shell/shell_material_inputs.hpp"

namespace litepgo::fem {

// Density per reference area: W = t*psi((a-a_bar)/2) + t^3/12*psi(b-b_bar),
// psi(S) = lambda_ps/2*tr(a_bar^-1*S)^2 + mu*tr(a_bar^-1*S*a_bar^-1*S).
// theta = [lambda_ps, mu] uses 2D plane-stress Lame parameters, not 3D lambda.
// Requires finite symmetric forms, t > 0, mu > 0, lambda_ps+mu > 0 (invalid_argument).
// Non-positive-definite metrics or unrepresentable results throw std::domain_error.
// Derivatives use (a11,a12,a22,b11,b12,b22); mixed entries are independent, unscaled.
class KoiterStVK final {
  public:
    static constexpr int num_parameters = 2;

    static double eval_W(const ShellMaterialInputs& inputs);
    static Eigen::Matrix<double, 6, 1> eval_dW_dab(const ShellMaterialInputs& inputs);
    static Eigen::Matrix<double, 6, 6> eval_d2W_dab2(const ShellMaterialInputs& inputs);
    // Rows: current (a11,a12,a22,b11,b12,b22); columns: natural forms in the same order.
    static Eigen::Matrix<double, 6, 6> eval_d2W_dab_dabarbbar(const ShellMaterialInputs& inputs);
    // Columns follow theta = [lambda_ps, mu]; geometry and thickness stay fixed.
    static Eigen::Matrix<double, 6, 2> eval_d2W_dab_dtheta(const ShellMaterialInputs& inputs);
};

} // namespace litepgo::fem
