#pragma once

#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/2d/tensor_product_basis_2d.hpp"

namespace litepgo::fem {

// Bicubic Hermite; corner order (0,0), (1,0), (1,1), (0,1).
// Per corner: u, u_xi, u_eta, u_xi_eta, with reference-coordinate derivatives.
using QuadHermiteBasis = TensorProductBasis2D<CubicHermiteBasis>;

} // namespace litepgo::fem
