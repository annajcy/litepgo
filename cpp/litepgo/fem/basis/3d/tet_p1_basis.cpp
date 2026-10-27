#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"

#include <stdexcept>

namespace litepgo {

TetP1Basis::Sample TetP1Basis::evaluate(const Eigen::Vector3d& xi) {
    Sample result;
    result.N << 1 - xi.sum(), xi.x(), xi.y(), xi.z();
    result.dN_dxi << -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    return result;
}

Eigen::Vector3d TetP1Basis::reference_node(int basis_index) {
    if (basis_index < 0 || basis_index >= num_basis_functions)
        throw std::out_of_range("TetP1Basis::reference_node: basis index is out of range");
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    if (basis_index > 0)
        result[basis_index - 1] = 1;
    return result;
}

} // namespace litepgo
