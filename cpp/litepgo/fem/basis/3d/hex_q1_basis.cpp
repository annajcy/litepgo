#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr std::array<std::array<int, 3>, 8> hex_corners{{
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1},
    {0, 1, 1},
}};

// libpgo referenceHex.h: Hex corner -> last-axis-fastest tensor index.
constexpr std::array<int, 8> hex_to_tensor{0, 4, 6, 2, 1, 5, 7, 3};

} // namespace

namespace litepgo::fem {

Eigen::Matrix<double, 8, 1> HexQ1Basis::eval_N(const Eigen::Vector3d& xi) {
    return TensorProductBasis3D<LinearLagrangeBasis>::eval_N(xi)(hex_to_tensor);
}

Eigen::Matrix<double, 8, 3> HexQ1Basis::eval_dN_dxi(const Eigen::Vector3d& xi) {
    return TensorProductBasis3D<LinearLagrangeBasis>::eval_dN_dxi(xi)(hex_to_tensor, Eigen::placeholders::all);
}

Eigen::Matrix<double, 8, 6> HexQ1Basis::eval_d2N_dxi2(const Eigen::Vector3d& xi) {
    return TensorProductBasis3D<LinearLagrangeBasis>::eval_d2N_dxi2(xi)(hex_to_tensor, Eigen::placeholders::all);
}

Eigen::Vector3d HexQ1Basis::reference_node(int basis_index) {
    if (basis_index < 0 || basis_index >= 8) {
        throw std::out_of_range("HexQ1Basis::reference_node: basis index is out of range");
    }
    const auto& corner = hex_corners[static_cast<std::size_t>(basis_index)];
    return {static_cast<double>(corner[0]), static_cast<double>(corner[1]), static_cast<double>(corner[2])};
}

} // namespace litepgo::fem
