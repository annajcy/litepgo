#pragma once

#include <Eigen/Core>

namespace litepgo {

// Geometry and basis data at one parameter location in the reference element.
// Coefficients consuming this point must use the same basis order and geometry.
struct VolumePointData {
    Eigen::Vector3d position;
    Eigen::Matrix3d jacobian; // dX/dxi
    Eigen::VectorXd basis_values;
    Eigen::Matrix<double, Eigen::Dynamic, 3> basis_gradients; // Row i is dN_i/dX.
    double jacobian_determinant;
};

} // namespace litepgo
