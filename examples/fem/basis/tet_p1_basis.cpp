#include <memory>
// Evaluate Tet P1 and reconstruct u(r,s,t) = 2*r + 3*s - t with a field.
// Build: cmake --build build/release --target tet_p1_basis
// Run: build/release/examples/fem/basis/tet_p1_basis
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

double target(const Eigen::Vector3d& xi) {
    return 2 * xi.x() + 3 * xi.y() - xi.z();
}

int main() {
    using B = litepgo::fem::TetP1Basis;
    static_assert(B::num_basis_functions == 4);
    const Eigen::Vector3d xi(0.25, 0.25, 0.25);
    const auto N = B::eval_N(xi);
    const auto dN = B::eval_dN_dxi(xi);
    const auto d2N = B::eval_d2N_dxi2(xi);

    Eigen::Matrix<double, 4, 3> expected_dN;
    expected_dN << -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    if (!N.isApprox(Eigen::Vector4d::Constant(0.25)) || !dN.isApprox(expected_dN) || !d2N.isZero())
        throw std::runtime_error("Tet P1 basis example check failed");
    std::cout << "TetP1: functions=" << B::num_basis_functions << ", verified=1\n";

    // Tet P1 coefficients are target values sampled at its four vertices.
    Eigen::MatrixXd coefficients(B::num_basis_functions, 1);
    for (int i = 0; i < B::num_basis_functions; ++i)
        coefficients(i, 0) = target(B::coefficient_info(i).point);
    litepgo::fem::InterpolatedResponse<B> field(coefficients);
    const std::array<Eigen::Vector3d, 3> points{Eigen::Vector3d(0.25, 0.25, 0.25), Eigen::Vector3d(0.2, 0.3, 0.1),
                                                Eigen::Vector3d(0.1, 0.1, 0.2)};
    for (const auto& point : points)
        if (std::abs(field.eval_value(point)[0] - target(point)) > 1e-12)
            throw std::runtime_error("Tet P1 field does not reconstruct the target");
    std::cout << "TetP1 field: samples=" << points.size() << ", verified=1\n";
}
