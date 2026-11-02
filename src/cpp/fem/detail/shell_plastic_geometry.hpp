#pragma once
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem::detail {
// The same Fp acts on the entire face stencil; translation has no effect on its forms.
inline TriangleMidEdgeKinematics shell_plastic_geometry(const Eigen::MatrixXd& reference,
                                                        const std::array<int, 3>& opposite, const Eigen::Matrix3d& Fp,
                                                        Eigen::MatrixXd* derivative = nullptr) {
    if (reference.rows() < 3 || reference.rows() > 6 || reference.cols() != 3 || !reference.allFinite())
        throw std::invalid_argument("Shell plastic geometry: invalid reference stencil");
    const double determinant = Fp.determinant();
    if (!Fp.allFinite() || !std::isfinite(determinant) || determinant <= 0)
        throw std::domain_error("Shell plastic geometry requires finite det(Fp) > 0");
    const Eigen::MatrixXd y = (reference.rowwise() - reference.row(0)) * Fp.transpose();
    if (!y.allFinite())
        throw std::domain_error("Shell plastic geometry: non-finite natural stencil");
    return eval_triangle_mid_edge_kinematics(y, opposite, derivative);
}
} // namespace litepgo::fem::detail
