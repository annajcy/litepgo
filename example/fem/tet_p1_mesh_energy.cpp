#include "tet_p1_mesh_energy.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace litepgo;

int main() {
    const Vec X = example::tet_p1_rest_positions();
    auto energy = example::make_tet_p1_mesh_energy(X);
    const Vec u = 0.1 * X; // Physical displacement u = 0.1 X, so F = 1.1 I in both elements.
    Vec gradient;
    SpMat hessian = energy.hessian_pattern();
    const double value = energy.evaluate(u, &gradient, &hessian);
    assert(energy.value(Vec::Zero(X.size())) == 0.0);
    assert(std::abs(value - 0.5 * NeoHookean::value({1.1 * Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(),
                                                     Eigen::Vector2d(2, 3)})) < 1e-12);
    std::cout << "E=" << value << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
