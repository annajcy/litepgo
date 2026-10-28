#include "litepgo/fem/kinematics/hex_hermite_dof_transform.hpp"

#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"

#include <array>
#include <stdexcept>

namespace litepgo {

Eigen::DiagonalMatrix<double, Eigen::Dynamic> eval_hex_hermite_dof_transform(const Eigen::MatrixXd& X_nodes) {
    if (X_nodes.rows() != 8 || X_nodes.cols() != 3 || !X_nodes.allFinite())
        throw std::invalid_argument("eval_hex_hermite_dof_transform requires finite 8-by-3 reference nodes");
    const Eigen::Vector3d origin = X_nodes.row(0).transpose();
    const Eigen::Vector3d lengths(X_nodes(1, 0) - origin.x(), X_nodes(3, 1) - origin.y(), X_nodes(4, 2) - origin.z());
    if (!lengths.allFinite() || (lengths.array() <= 0).any())
        throw std::invalid_argument("eval_hex_hermite_dof_transform requires positive axis-aligned box dimensions");
    for (int v = 0; v < 8; ++v) {
        const Eigen::Vector3d normalized = (X_nodes.row(v).transpose() - origin).cwiseQuotient(lengths);
        if (!normalized.allFinite() || !(normalized - HexHermiteBasis::reference_node(8 * v)).isZero(1e-12))
            throw std::invalid_argument(
                "eval_hex_hermite_dof_transform requires axis-aligned boxes in Hex corner order");
    }
    const auto [hx, hy, hz] = std::array{lengths.x(), lengths.y(), lengths.z()};
    const Eigen::Matrix<double, 8, 1> slots{1, hx, hy, hz, hx * hy, hx * hz, hy * hz, hx * hy * hz};
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> S(3 * HexHermiteBasis::num_basis_functions);
    auto& scales = S.diagonal();
    for (int v = 0; v < 8; ++v)
        for (int slot = 0; slot < 8; ++slot)
            scales.segment<3>(3 * (8 * v + slot)).setConstant(slots[slot]);
    if (!scales.allFinite() || scales.minCoeff() <= 0)
        throw std::invalid_argument("eval_hex_hermite_dof_transform requires finite positive DOF scales");
    return S;
}

} // namespace litepgo
