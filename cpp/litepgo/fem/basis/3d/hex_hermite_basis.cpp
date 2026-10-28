#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"

#include <array>
#include <stdexcept>

namespace {

template <int Columns>
Eigen::Matrix<double, 64, Columns> vertex_order(const Eigen::Matrix<double, 64, Columns>& tensor) {
    // Derivative DOF slots add 16, 4, or 1 to the tensor index (i*4+j)*4+k.
    constexpr std::array<int, 8> offsets{0, 16, 4, 1, 20, 17, 5, 21};
    Eigen::Matrix<double, 64, Columns> result;
    for (int vertex = 0; vertex < 8; ++vertex) {
        const auto corner = litepgo::HexQ1Basis::reference_node(vertex);
        const int base =
            32 * static_cast<int>(corner.x()) + 8 * static_cast<int>(corner.y()) + 2 * static_cast<int>(corner.z());
        for (int slot = 0; slot < 8; ++slot) {
            const int index = base + offsets[static_cast<std::size_t>(slot)];
            result.row(8 * vertex + slot) = tensor.row(index);
        }
    }
    return result;
}

} // namespace

namespace litepgo {

Eigen::Matrix<double, 64, 1> HexHermiteBasis::evaluate_N(const Eigen::Vector3d& xi) {
    return vertex_order(TensorProductBasis3D<CubicHermiteBasis>::evaluate_N(xi));
}

Eigen::Matrix<double, 64, 3> HexHermiteBasis::evaluate_dN_dxi(const Eigen::Vector3d& xi) {
    return vertex_order(TensorProductBasis3D<CubicHermiteBasis>::evaluate_dN_dxi(xi));
}

Eigen::Matrix<double, 64, 6> HexHermiteBasis::evaluate_d2N_dxi2(const Eigen::Vector3d& xi) {
    return vertex_order(TensorProductBasis3D<CubicHermiteBasis>::evaluate_d2N_dxi2(xi));
}

Eigen::Vector3d HexHermiteBasis::reference_node(int basis_index) {
    if (basis_index < 0 || basis_index >= num_basis_functions)
        throw std::out_of_range("HexHermiteBasis::reference_node: basis index is out of range");
    return HexQ1Basis::reference_node(basis_index / 8);
}

} // namespace litepgo
