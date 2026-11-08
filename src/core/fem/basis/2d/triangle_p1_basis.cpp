#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"

#include <stdexcept>

namespace litepgo::fem {

Eigen::Vector3d TriangleP1Basis::eval_N(const Eigen::Vector2d& xi) {
    return {1 - xi.sum(), xi.x(), xi.y()};
}

Eigen::Matrix<double, 3, 2> TriangleP1Basis::eval_dN_dxi(const Eigen::Vector2d&) {
    Eigen::Matrix<double, 3, 2> result;
    result << -1, -1, 1, 0, 0, 1;
    return result;
}

Eigen::Matrix<double, 3, 3> TriangleP1Basis::eval_d2N_dxi2(const Eigen::Vector2d&) {
    return Eigen::Matrix<double, 3, 3>::Zero();
}

CoefficientInfo<2> TriangleP1Basis::coefficient_info(int basis_index) {
    if (basis_index < 0 || basis_index >= num_basis_functions)
        throw std::out_of_range("TriangleP1Basis::coefficient_info: basis index is out of range");
    Eigen::Vector2d result = Eigen::Vector2d::Zero();
    if (basis_index > 0)
        result[basis_index - 1] = 1;
    return {result, {}};
}

} // namespace litepgo::fem
