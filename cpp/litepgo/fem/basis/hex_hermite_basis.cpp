#include "litepgo/fem/basis/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"

#include <array>
#include <stdexcept>

namespace litepgo {

HexHermiteBasis::Sample HexHermiteBasis::evaluate(const Eigen::Vector3d& xi) {
    const auto tensor = TensorProductBasis3D<CubicHermiteBasis>::evaluate(xi);
    // Derivative DOF slots add 16, 4, or 1 to the tensor index (i*4+j)*4+k.
    constexpr std::array<int, 8> offsets{0, 16, 4, 1, 20, 17, 5, 21};
    Sample result;
    for (int vertex = 0; vertex < 8; ++vertex) {
        const auto corner = HexQ1Basis::reference_node(vertex);
        const int base =
            32 * static_cast<int>(corner.x()) + 8 * static_cast<int>(corner.y()) + 2 * static_cast<int>(corner.z());
        for (int slot = 0; slot < 8; ++slot) {
            const int index = base + offsets[static_cast<std::size_t>(slot)];
            result.N[8 * vertex + slot] = tensor.N[index];
            result.dN_dxi.row(8 * vertex + slot) = tensor.dN_dxi.row(index);
        }
    }
    return result;
}

Eigen::Vector3d HexHermiteBasis::reference_node(int basis_index) {
    if (basis_index < 0 || basis_index >= num_basis_functions)
        throw std::out_of_range("HexHermiteBasis::reference_node: basis index is out of range");
    return HexQ1Basis::reference_node(basis_index / 8);
}

} // namespace litepgo
