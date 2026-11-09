#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include <iostream>
#include <stdexcept>
int main() {
    using namespace litepgo;
    Eigen::MatrixXd X(4, 3);
    X << 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4;
    fem::InterpolatedResponse<fem::TetP1Basis> geometry(X);
    const Eigen::Vector3d xi(.2, .3, .1);
    auto evaluation = geometry.evaluate(xi);
    const auto J = evaluation->jacobian();
    const Vec value = evaluation->value();
    const Eigen::Matrix3d expected_J = Eigen::Vector3d(2, 3, 4).asDiagonal();
    if (!value.isApprox(Eigen::Vector3d(.4, .9, .4)) || !Eigen::MatrixXd(J).isApprox(expected_J) ||
        !Eigen::MatrixXd(geometry.evaluate(xi)->hessian(Eigen::Vector3d::Ones())).isZero())
        throw std::runtime_error("geometry mismatch");
    Eigen::MatrixXd C(8, 2);
    for (int a = 0; a < 8; ++a) {
        const auto p = fem::HexQ1Basis::coefficient_info(a).point;
        C(a, 0) = p.x() * p.y();
        C(a, 1) = p.y() * p.z();
    }
    fem::InterpolatedResponse<fem::HexQ1Basis> response(C);
    const auto H = response.evaluate(xi)->hessian(Eigen::Vector2d(2, -3));
    Eigen::Matrix3d expected_H;
    expected_H << 0, 2, 0, 2, 0, -3, 0, -3, 0;
    if (!Eigen::MatrixXd(H).isApprox(expected_H))
        throw std::runtime_error("weighted Hessian mismatch");
    std::cout << "spatial geometry and weighted Hessian: verified\n";
}
