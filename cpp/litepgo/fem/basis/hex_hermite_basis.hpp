#pragma once

#include "litepgo/fem/basis/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/tensor_product_basis.hpp"

namespace litepgo {

// Tricubic Hermite basis on [0,1]^3, in HexQ1Basis corner order.
// Index 8*vertex + slot uses slots u, u_xi, u_eta, u_zeta,
// u_xi_eta, u_xi_zeta, u_eta_zeta, u_xi_eta_zeta.
// Coefficients are reference-parameter derivatives, without physical length scaling.
class HexHermiteBasis final {
  public:
    static constexpr int num_basis_functions = 64;
    using Sample = TensorProductBasis3D<CubicHermiteBasis>::Sample;

    // N and dN_dxi rows use the same slot order; derivatives are with respect
    // to xi, eta, zeta. Extrapolates without clamping.
    static Sample evaluate(const Eigen::Vector3d& xi);

    // Reference corner of the basis function, repeated for each of its eight slots.
    // Invalid basis indices throw std::out_of_range.
    static Eigen::Vector3d reference_node(int basis_index);
};

} // namespace litepgo
