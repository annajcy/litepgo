#pragma once

#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/geometry/interpolation.hpp"

namespace litepgo {

// Returns diagonal S in c = S * flatten(x_local), with 192 scalar DOFs for a 3D vector field.
// x_local is 8-by-24: each vertex stores u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ, each a 3-vector.
// c reshapes by row into the 64-by-3 reference coefficients expected by HexHermiteBasis.
// Compute once per fixed rest element. Invalid geometry or non-finite/non-positive scales throw invalid_argument.
// shortcut: positive axis-aligned rest boxes in Q1 corner order only; general rest hexes need a different mapping.
Eigen::DiagonalMatrix<double, Eigen::Dynamic> hex_hermite_dof_transform(const Interpolation<HexQ1Basis>& geometry);

} // namespace litepgo
