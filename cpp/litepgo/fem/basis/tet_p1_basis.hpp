#pragma once

#include <Eigen/Core>

namespace litepgo {

// Scalar P1 basis on r,s,t >= 0, r+s+t <= 1, with vertices ordered as
// (0,0,0), (1,0,0), (0,1,0), (0,0,1).
class TetP1Basis final {
  public:
    static constexpr int num_basis_functions = 4;
    struct Sample {
        Eigen::Vector4d N;
        Eigen::Matrix<double, 4, 3> dN_dxi;
    };

    // Gradients are with respect to parameter coordinates; extrapolates without clamping.
    static Sample evaluate(const Eigen::Vector3d& xi);

    // Invalid basis indices throw std::out_of_range.
    static Eigen::Vector3d reference_node(int basis_index);
};

} // namespace litepgo
