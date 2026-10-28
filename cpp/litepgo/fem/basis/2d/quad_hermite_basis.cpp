#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"

#include <array>
#include <stdexcept>

namespace {

template <int Columns>
Eigen::Matrix<double, 16, Columns> vertex_order(const Eigen::Matrix<double, 16, Columns>& tensor) {
    constexpr std::array<int, 4> offsets{0, 4, 1, 5};
    Eigen::Matrix<double, 16, Columns> result;
    for (int vertex = 0; vertex < 4; ++vertex) {
        const auto corner = litepgo::QuadHermiteBasis::reference_node(4 * vertex);
        const int base = 8 * static_cast<int>(corner.x()) + 2 * static_cast<int>(corner.y());
        for (int slot = 0; slot < 4; ++slot) {
            const int row = 4 * vertex + slot;
            const int index = base + offsets[static_cast<std::size_t>(slot)];
            result.row(row) = tensor.row(index);
        }
    }
    return result;
}

} // namespace

namespace litepgo {

Eigen::Matrix<double, 16, 1> QuadHermiteBasis::evaluate_N(const Eigen::Vector2d& xi) {
    return vertex_order(TensorProductBasis2D<CubicHermiteBasis>::evaluate_N(xi));
}

Eigen::Matrix<double, 16, 2> QuadHermiteBasis::evaluate_dN_dxi(const Eigen::Vector2d& xi) {
    return vertex_order(TensorProductBasis2D<CubicHermiteBasis>::evaluate_dN_dxi(xi));
}

Eigen::Matrix<double, 16, 3> QuadHermiteBasis::evaluate_d2N_dxi2(const Eigen::Vector2d& xi) {
    return vertex_order(TensorProductBasis2D<CubicHermiteBasis>::evaluate_d2N_dxi2(xi));
}

Eigen::Vector2d QuadHermiteBasis::reference_node(int basis_index) {
    if (basis_index < 0 || basis_index >= num_basis_functions)
        throw std::out_of_range("QuadHermiteBasis::reference_node: basis index is out of range");
    constexpr std::array<std::array<int, 2>, 4> corners{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    const auto& corner = corners[static_cast<std::size_t>(basis_index / 4)];
    return {static_cast<double>(corner[0]), static_cast<double>(corner[1])};
}

} // namespace litepgo
