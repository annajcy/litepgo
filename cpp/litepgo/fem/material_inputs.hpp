#pragma once

#include <Eigen/Core>

namespace litepgo {

// Owns the pointwise inputs; no geometry, interpolation, or DOF layout.
template <int NumParameters>
    requires(NumParameters > 0)
struct MaterialInputs {
    Eigen::Matrix3d F;
    Eigen::Matrix3d Fp;
    Eigen::Matrix<double, NumParameters, 1> theta;
};

} // namespace litepgo
