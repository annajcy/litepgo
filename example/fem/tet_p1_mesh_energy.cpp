#include "tet_p1_mesh_energy.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace litepgo;

int main() {
    const Vec X_nodes = example::tet_p1_rest_positions();
    auto energy = example::make_tet_p1_mesh_energy(X_nodes);
    const Vec u = 0.1 * X_nodes; // Physical displacement u = 0.1 X, so F = 1.1 I in both elements.
    Vec gradient;
    SpMat hessian = energy.hessian_pattern();
    const double E = energy.eval(u, &gradient, &hessian);
    assert(energy.eval_value(Vec::Zero(X_nodes.size())) == 0.0);
    assert(std::abs(E - 0.5 * NeoHookean::eval_W({1.1 * Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(),
                                                  Eigen::Vector2d(2, 3)})) < 1e-12);
    std::cout << "E=" << E << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
