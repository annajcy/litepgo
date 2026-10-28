#pragma once

#include <Eigen/Core>
#include <array>

namespace litepgo {

struct TriangleMidEdgeKinematics {
    Eigen::Matrix2d a;
    Eigen::Matrix2d b;
};

// x has 3 to 6 rows: the central face first, then unique opposite vertices.
// opposite[k] is the local row across edge k, or -1 at a boundary (zero edge curvature).
// Uses area-weighted mid-edge normals; b follows the convention b_ab = n.dot(x_,ab).
// Optional outputs are independently nullable and overwritten. Rows/components use
// (a11,a12,a22,b11,b12,b22), without doubling mixed entries; x flattens by row.
// dab_dx is 6-by-3n; each d2ab_dx2 entry is 3n-by-3n.
// Invalid layout/non-finite inputs throw invalid_argument; degenerate faces,
// cancelling adjacent normals, or unrepresentable results throw InfeasibleState.
TriangleMidEdgeKinematics eval_triangle_mid_edge_kinematics(const Eigen::Ref<const Eigen::MatrixXd>& x,
                                                            const std::array<int, 3>& opposite,
                                                            Eigen::MatrixXd* dab_dx = nullptr,
                                                            std::array<Eigen::MatrixXd, 6>* d2ab_dx2 = nullptr);

} // namespace litepgo
