#include "shell_stvk_mesh_energy.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace litepgo;

int main() {
    const Vec X_nodes = example::shell_stvk_rest_positions();
    auto energy = example::make_shell_stvk_mesh_energy(X_nodes);
    const Vec u = 0.1 * X_nodes; // Uniform stretch: a = 1.21*a_bar, b = b_bar = 0.
    Vec gradient;
    SpMat hessian = energy.hessian_pattern();
    const double E = energy.eval(u, &gradient, &hessian);
    assert(energy.eval_value(Vec::Zero(X_nodes.size())) == 0.0);
    const double Y = 100, nu = 0.3;
    const Eigen::Vector2d theta(Y * nu / (1 - nu * nu), Y / (2 * (1 + nu)));
    assert(std::abs(E - 5.5 * KoiterStVK::eval_W({1.21 * Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Zero(),
                                                  Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Zero(), 0.1, theta})) <
           1e-12);
    std::cout << "E=" << E << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
