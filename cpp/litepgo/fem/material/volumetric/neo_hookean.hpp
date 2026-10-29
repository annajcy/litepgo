#pragma once

#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"

namespace litepgo::fem {

// W = det(Fp) * psi, psi = mu/2 (||Fe||^2 - 3) - mu log(det_Fe) + lambda/2 log(det_Fe)^2,
// Fe = F Fp^-1, det_Fe = det(Fe). W is density per original reference volume.
// Requires finite theta = [lambda, mu], lambda >= 0, mu > 0,
// and finite F/Fp (invalid_argument otherwise). Non-positive det(Fp)/det(Fe)
// or unrepresentable kinematics, density, or requested derivatives throw std::domain_error.
// Derivatives hold the other arguments fixed. dW_dF/dW_dFp retain 3-by-3 matrix shape;
// their 9-by-9 Hessians differentiate column-major vec(F)/vec(Fp).
class NeoHookean final {
  public:
    static constexpr int num_parameters = 2;

    static double eval_W(const VolumetricMaterialInputs& inputs);
    static Eigen::Matrix3d eval_dW_dF(const VolumetricMaterialInputs& inputs);
    static Eigen::Matrix<double, 9, 9> eval_d2W_dF2(const VolumetricMaterialInputs& inputs);
    static Eigen::Matrix3d eval_dW_dFp(const VolumetricMaterialInputs& inputs);
    static Eigen::Matrix<double, 9, 9> eval_d2W_dFp2(const VolumetricMaterialInputs& inputs);
    static Eigen::Vector2d eval_dW_dtheta(const VolumetricMaterialInputs& inputs);
    static Eigen::Matrix2d eval_d2W_dtheta2(const VolumetricMaterialInputs& inputs);

    // Inputs vary affinely: inputs+t*delta_inputs. Inactive channels have zero
    // increments. Preserves det(F), det(Fp)>0 and lambda>=0, mu>0 on [0,alpha].
    static double eval_feasible_step(const VolumetricMaterialInputs& inputs,
                                     const VolumetricMaterialInputs& delta_inputs);
};

} // namespace litepgo::fem
