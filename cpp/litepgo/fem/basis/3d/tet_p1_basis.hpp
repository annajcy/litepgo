#pragma once

#include "litepgo/fem/basis/basis_sample.hpp"

namespace litepgo {

// Scalar P1 basis on r,s,t >= 0, r+s+t <= 1, with vertices ordered as
// (0,0,0), (1,0,0), (0,1,0), (0,0,1).
class TetP1Basis final {
  public:
    static constexpr int num_basis_functions = 4;
    using Sample = BasisSample<num_basis_functions, 3>;

    // Reference gradients are constant and second derivatives are zero. Extrapolates without clamping.
    static Sample evaluate(const Eigen::Vector3d& xi);

    // Invalid basis indices throw std::out_of_range.
    static Eigen::Vector3d reference_node(int basis_index);
};

} // namespace litepgo
