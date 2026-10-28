#pragma once

#include <Eigen/Core>

namespace litepgo {

// Returns diagonal S in c = S * flatten(x_local), with 192 scalar DOFs for a 3D vector field.
// x_local is 8-by-24: each vertex stores u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ, each a 3-vector.
// c reshapes by row into the 64-by-3 reference coefficients expected by HexHermiteBasis.
// X_nodes is 8-by-3; its rows and x_local rows follow HexHermiteBasis vertex order.
// Compute once per fixed rest element. Invalid dimensions, non-finite coordinates, non-box geometry,
// or non-finite/non-positive lengths or scales throw invalid_argument.
// shortcut: positive axis-aligned rest boxes only; general rest hexes need a different DOF mapping.
Eigen::DiagonalMatrix<double, Eigen::Dynamic> eval_hex_hermite_dof_transform(const Eigen::MatrixXd& X_nodes);

} // namespace litepgo
