#include "hex_hermite_mesh_energy.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace litepgo;

int main() {
    const Vec X_nodes = example::hex_hermite_rest_positions();
    auto energy = example::make_hex_hermite_mesh_energy(X_nodes);
    // Physical u = 0.1*X: first derivatives are 0.1*I; mixed derivatives are zero.
    Vec x = Vec::Zero(12 * 24);
    for (int v = 0; v < 12; ++v) {
        x.segment<3>(24 * v) = 0.1 * X_nodes.segment<3>(3 * v);
        for (int d = 0; d < 3; ++d)
            x.segment<3>(24 * v + 3 * (d + 1)) = 0.1 * Eigen::Vector3d::Unit(d);
    }
    Vec gradient;
    SpMat hessian = energy.hessian_pattern();
    const double E = energy.evaluate(x, &gradient, &hessian);
    assert(energy.value(Vec::Zero(x.size())) == 0.0);
    assert(std::abs(E - 18 * NeoHookean::value({1.1 * Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(),
                                                Eigen::Vector2d(2, 3)})) < 1e-11);
    std::cout << "E=" << E << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
