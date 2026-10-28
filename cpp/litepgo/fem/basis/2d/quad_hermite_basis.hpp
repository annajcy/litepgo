#pragma once

#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/2d/tensor_product_basis_2d.hpp"

namespace litepgo {

// Bicubic Hermite basis on [0,1]^2, corners (0,0), (1,0), (1,1), (0,1).
// Index 4*vertex+slot uses slots u, u_xi, u_eta, u_xi_eta.
// Coefficients are reference-parameter derivatives, without length scaling.
class QuadHermiteBasis final {
  public:
    static constexpr int num_basis_functions = 16;
    using Sample = TensorProductBasis2D<CubicHermiteBasis>::Sample;

    // Extrapolates without clamping. All arrays use the same vertex/slot row order.
    static Sample evaluate(const Eigen::Vector2d& xi);

    // The corner owning a basis function, repeated for each of its four slots.
    static Eigen::Vector2d reference_node(int basis_index);
};

} // namespace litepgo
