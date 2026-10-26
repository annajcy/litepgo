#include "litepgo/fem/interpolation/hex_hermite_dof_transform.hpp"

#include "litepgo/fem/basis/hex_hermite_basis.hpp"
#include "litepgo/fem/interpolation/rest_point.hpp"

#include <array>
#include <stdexcept>

namespace litepgo {

Eigen::DiagonalMatrix<double, Eigen::Dynamic>
hex_hermite_dof_transform(const Interpolation<HexQ1Basis>& rest_geometry) {
    const Eigen::Vector3d lengths = rest_point(rest_geometry, Eigen::Vector3d::Zero()).dX_dxi.diagonal();
    if ((lengths.array() <= 0).any())
        throw std::invalid_argument("hex_hermite_dof_transform requires positive axis-aligned box dimensions");
    for (int v = 0; v < 8; ++v) {
        const Eigen::Matrix3d normalized =
            rest_geometry.dvalue_dxi(HexQ1Basis::reference_node(v)) * lengths.cwiseInverse().asDiagonal();
        if (!normalized.isIdentity(1e-12))
            throw std::invalid_argument("hex_hermite_dof_transform requires axis-aligned boxes in Hex corner order");
    }
    const auto [hx, hy, hz] = std::array{lengths.x(), lengths.y(), lengths.z()};
    const Eigen::Matrix<double, 8, 1> slots{1, hx, hy, hz, hx * hy, hx * hz, hy * hz, hx * hy * hz};
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> S(3 * HexHermiteBasis::num_basis_functions);
    auto& scales = S.diagonal();
    for (int v = 0; v < 8; ++v)
        for (int slot = 0; slot < 8; ++slot)
            scales.segment<3>(3 * (8 * v + slot)).setConstant(slots[slot]);
    if (!scales.allFinite() || scales.minCoeff() <= 0)
        throw std::invalid_argument("hex_hermite_dof_transform requires finite positive DOF scales");
    return S;
}

} // namespace litepgo
