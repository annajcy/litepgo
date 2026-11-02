#pragma once

#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
#include "litepgo/fem/basis/3d/tensor_product_basis_3d.hpp"

namespace litepgo::fem {

// Scalar Q1 basis on [0,1]^3, in libpgo Hex corner order:
// (0,0,0), (1,0,0), (1,1,0), (0,1,0),
// (0,0,1), (1,0,1), (1,1,1), (0,1,1).
class HexQ1Basis final {
  public:
    static constexpr int num_basis_functions = 8;
    // All rows follow Hex corner order; derivatives use parameter coordinates.
    // Outside [0,1]^3, extrapolates without clamping.
    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector3d& xi);
    static Eigen::Matrix<double, num_basis_functions, 3> eval_dN_dxi(const Eigen::Vector3d& xi);
    static Eigen::Matrix<double, num_basis_functions, 6> eval_d2N_dxi2(const Eigen::Vector3d& xi);

    // Reference interpolation node for the given basis function.
    // Invalid basis indices throw std::out_of_range.
    static Eigen::Vector3d reference_node(int basis_index);
};

} // namespace litepgo::fem
