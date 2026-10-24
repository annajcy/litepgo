#pragma once

#include "litepgo/fem/lagrange_basis.hpp"
#include "litepgo/fem/tensor_product_basis.hpp"

namespace litepgo {

// Scalar Q1 basis on [0,1]^3, in libpgo Hex corner order:
// (0,0,0), (1,0,0), (1,1,0), (0,1,0),
// (0,0,1), (1,0,1), (1,1,1), (0,1,1).
class HexQ1Basis final {
  public:
    static constexpr int num_basis_functions = 8;
    using Sample = TensorProductBasis3D<LagrangeBasis>::Sample;

    // Values and gradient rows both follow Hex corner order. Gradients are with
    // respect to parameter coordinates. Outside [0,1]^3, extrapolates without clamping.
    static Sample evaluate(const Eigen::Vector3d& xi);

    // Reference interpolation node for the given basis function.
    // Invalid basis indices throw std::out_of_range.
    static Eigen::Vector3d reference_node(int basis_index);
};

} // namespace litepgo
