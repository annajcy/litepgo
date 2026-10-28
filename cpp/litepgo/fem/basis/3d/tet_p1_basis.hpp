#pragma once

#include <Eigen/Core>

namespace litepgo {

// Scalar P1 basis on r,s,t >= 0, r+s+t <= 1, with vertices ordered as
// (0,0,0), (1,0,0), (0,1,0), (0,0,1).
class TetP1Basis final {
  public:
    static constexpr int num_basis_functions = 4;
    // Reference gradients are constant and second derivatives are zero. Extrapolates without clamping.
    static Eigen::Vector4d evaluate_N(const Eigen::Vector3d& xi);
    static Eigen::Matrix<double, num_basis_functions, 3> evaluate_dN_dxi(const Eigen::Vector3d& xi);
    static Eigen::Matrix<double, num_basis_functions, 6> evaluate_d2N_dxi2(const Eigen::Vector3d& xi);

    // Invalid basis indices throw std::out_of_range.
    static Eigen::Vector3d reference_node(int basis_index);
};

} // namespace litepgo
