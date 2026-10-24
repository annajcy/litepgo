#pragma once
#include <Eigen/Core>

namespace litepgo {
// Pointwise constitutive inputs; parameter count is validated by each material.
struct MaterialInputs {
    Eigen::Matrix3d F;
    Eigen::Matrix3d Fp;
    Eigen::VectorXd theta;
};
} // namespace litepgo
