#pragma once

#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/tensor_product_basis_3d.hpp"

namespace litepgo::fem {

// Tricubic Hermite in Hex corner order. Per corner: u, u_xi, u_eta, u_zeta,
// u_xi_eta, u_xi_zeta, u_eta_zeta, u_xi_eta_zeta; no physical length scaling.
using HexHermiteBasis = TensorProductBasis3D<CubicHermiteBasis>;

} // namespace litepgo::fem
