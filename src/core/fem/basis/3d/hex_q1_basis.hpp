#pragma once

#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
#include "litepgo/fem/basis/3d/tensor_product_basis_3d.hpp"

namespace litepgo::fem {

// Trilinear basis in Hex corner order: (0,0,0), (1,0,0), (1,1,0), (0,1,0),
// (0,0,1), (1,0,1), (1,1,1), (0,1,1).
using HexQ1Basis = TensorProductBasis3D<LinearLagrangeBasis>;

} // namespace litepgo::fem
