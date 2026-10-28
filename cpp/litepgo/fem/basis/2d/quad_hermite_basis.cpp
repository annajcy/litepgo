#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"

#include <array>
#include <stdexcept>

namespace litepgo {

QuadHermiteBasis::Sample QuadHermiteBasis::evaluate(const Eigen::Vector2d& xi) {
    const auto tensor = TensorProductBasis2D<CubicHermiteBasis>::evaluate(xi);
    constexpr std::array<int, 4> offsets{0, 4, 1, 5};
    Sample result;
    for (int vertex = 0; vertex < 4; ++vertex) {
        const auto corner = reference_node(4 * vertex);
        const int base = 8 * static_cast<int>(corner.x()) + 2 * static_cast<int>(corner.y());
        for (int slot = 0; slot < 4; ++slot) {
            const int row = 4 * vertex + slot;
            const int index = base + offsets[static_cast<std::size_t>(slot)];
            result.N[row] = tensor.N[index];
            result.dN_dxi.row(row) = tensor.dN_dxi.row(index);
            result.d2N_dxi2.row(row) = tensor.d2N_dxi2.row(index);
        }
    }
    return result;
}

Eigen::Vector2d QuadHermiteBasis::reference_node(int basis_index) {
    if (basis_index < 0 || basis_index >= num_basis_functions)
        throw std::out_of_range("QuadHermiteBasis::reference_node: basis index is out of range");
    constexpr std::array<std::array<int, 2>, 4> corners{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    const auto& corner = corners[static_cast<std::size_t>(basis_index / 4)];
    return {static_cast<double>(corner[0]), static_cast<double>(corner[1])};
}

} // namespace litepgo
