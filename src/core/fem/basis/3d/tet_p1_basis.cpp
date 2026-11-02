#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"

#include <stdexcept>

namespace litepgo::fem {

Eigen::Vector4d TetP1Basis::eval_N(const Eigen::Vector3d& xi) {
    return {1 - xi.sum(), xi.x(), xi.y(), xi.z()};
}

Eigen::Matrix<double, 4, 3> TetP1Basis::eval_dN_dxi(const Eigen::Vector3d&) {
    Eigen::Matrix<double, 4, 3> result;
    result << -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    return result;
}

Eigen::Matrix<double, 4, 6> TetP1Basis::eval_d2N_dxi2(const Eigen::Vector3d&) {
    return Eigen::Matrix<double, 4, 6>::Zero();
}

Eigen::Vector3d TetP1Basis::reference_node(int basis_index) {
    if (basis_index < 0 || basis_index >= num_basis_functions)
        throw std::out_of_range("TetP1Basis::reference_node: basis index is out of range");
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    if (basis_index > 0)
        result[basis_index - 1] = 1;
    return result;
}

} // namespace litepgo::fem
