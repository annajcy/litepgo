#pragma once

#include "litepgo/fem/basis/basis.hpp"

namespace litepgo::fem {

// Scalar P1 basis on xi,eta >= 0, xi+eta <= 1, with vertices ordered as
// (0,0), (1,0), (0,1).
class TriangleP1Basis final {
  public:
    static constexpr int num_basis_functions = 3;
    // Reference gradients are constant and second derivatives are zero. Extrapolates without clamping.
    static Eigen::Vector3d eval_N(const Eigen::Vector2d& xi);
    static Eigen::Matrix<double, num_basis_functions, 2> eval_dN_dxi(const Eigen::Vector2d& xi);
    static Eigen::Matrix<double, num_basis_functions, 3> eval_d2N_dxi2(const Eigen::Vector2d& xi);

    // Coefficients sample values at vertices; invalid indices throw std::out_of_range.
    static CoefficientInfo<2> coefficient_info(int basis_index);
};

} // namespace litepgo::fem
