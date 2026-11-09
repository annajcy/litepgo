// Fixed-point Triangle P1 interpolation composed with a positive nonlinear response.
#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/exp_response.hpp"
#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace litepgo;

int main() {
    const Eigen::Vector2d xi(0.2, 0.3);
    const Eigen::Vector3d c(-0.2, 0.1, 0.4);
    const Eigen::Vector3d N = fem::TriangleP1Basis::eval_N(xi);
    const auto sample = std::make_shared<energy::AffineResponse>(SpMat(N.transpose().sparseView()));
    energy::ComposedResponse response(std::make_shared<energy::ExpResponse>(1), sample);
    const double theta = response.evaluate(c)->value()[0];
    const Eigen::MatrixXd J = response.evaluate(c)->jacobian();
    const Eigen::MatrixXd H = response.evaluate(c)->hessian(Vec::Ones(1));
    const Eigen::Matrix3d expected = theta * N * N.transpose();
    if (!H.isApprox(expected, 1e-12))
        throw std::runtime_error("Incorrect nonlinear response Hessian");

    // W(theta) = theta^2 / 2: W' = theta, W'' = 1. Both chain-rule terms are needed.
    const Eigen::Matrix3d energy_H =
        J.transpose() * J + Eigen::MatrixXd(response.evaluate(c)->hessian(Vec::Constant(1, theta)));
    if (!energy_H.isApprox(2 * theta * theta * N * N.transpose(), 1e-12))
        throw std::runtime_error("Incorrect composed energy Hessian");
    std::cout << std::fixed << std::setprecision(6) << "theta=" << theta << ", response H00=" << H(0, 0)
              << ", energy H00=" << energy_H(0, 0) << '\n';
}
