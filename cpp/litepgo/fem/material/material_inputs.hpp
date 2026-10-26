#pragma once
#include <Eigen/Core>

namespace litepgo {
// Pointwise constitutive inputs; parameter count is validated by each material.
struct MaterialInputs {
    Eigen::Matrix3d F;     // Total deformation gradient d(x_position)/dX; x_position = X + u(X).
    Eigen::Matrix3d Fp;    // Plastic part of F = Fe * Fp; Fe is the elastic deformation gradient.
    Eigen::VectorXd theta; // Model-specific parameters; NeoHookean uses [lambda, mu].
};
} // namespace litepgo
