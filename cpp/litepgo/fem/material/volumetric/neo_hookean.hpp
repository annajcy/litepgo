#pragma once

#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"

namespace litepgo::fem {

// W = det(Fp) * psi, psi = mu/2 (||Fe||^2 - 3) - mu log(det_Fe) + lambda/2 log(det_Fe)^2,
// Fe = F Fp^-1, det_Fe = det(Fe). W is density per original reference volume.
// Requires finite theta = [lambda, mu], lambda >= 0, mu > 0,
// and finite F/Fp (invalid_argument otherwise). Non-positive det(Fp)/det(Fe)
// or unrepresentable kinematics, density, or requested derivatives throw std::domain_error.
// Derivatives hold the other arguments fixed. dW_dF retains 3-by-3 matrix shape;
// its 9-by-9 Hessian differentiates column-major vec(F).
class NeoHookean final {
  public:
    static constexpr int num_parameters = 2;

    static double eval_W(const VolumetricMaterialInputs& inputs);
    static Eigen::Matrix3d eval_dW_dF(const VolumetricMaterialInputs& inputs);
    static Eigen::Matrix<double, 9, 9> eval_d2W_dF2(const VolumetricMaterialInputs& inputs);

    // d vec(dW/dF) / d theta at fixed F and Fp. Rows use i+3*j;
    // columns follow theta=[lambda, mu]. Same domain checks as eval_dW_dF.
    static Eigen::Matrix<double, 9, num_parameters> eval_d2W_dF_dtheta(const VolumetricMaterialInputs& inputs);

    // d vec(dW/dF) / d vec(Fp), at fixed F and theta; both use i+3*j.
    static Eigen::Matrix<double, 9, 9> eval_d2W_dF_dFp(const VolumetricMaterialInputs& inputs);

    // Inputs vary affinely: inputs+t*delta_inputs. Inactive channels have zero
    // increments. Preserves det(F), det(Fp)>0 and lambda>=0, mu>0 on [0,alpha].
    static double eval_feasible_step(const VolumetricMaterialInputs& inputs,
                                     const VolumetricMaterialInputs& delta_inputs);
};

} // namespace litepgo::fem
