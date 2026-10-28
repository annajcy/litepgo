#pragma once

#include <Eigen/Core>

namespace litepgo {

// Linear nodal basis on [0, 1], ordered by the nodes 0, 1 (libpgo convention).
class LinearLagrangeBasis {
  public:
    static constexpr int num_basis_functions = 2;

    // Evaluates the polynomials without clamping the parameter to the interval.
    static Eigen::Vector2d eval_N(double t) {
        return {1.0 - t, t};
    }
    static Eigen::Vector2d eval_dN_dxi(double) {
        return {-1.0, 1.0};
    }
    static Eigen::Vector2d eval_d2N_dxi2(double) {
        return Eigen::Vector2d::Zero();
    }
};

} // namespace litepgo
