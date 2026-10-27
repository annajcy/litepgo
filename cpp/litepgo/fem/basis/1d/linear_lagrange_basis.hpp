#pragma once

#include <Eigen/Core>

namespace litepgo {

// Linear nodal basis on [0, 1], ordered by the nodes 0, 1 (libpgo convention).
class LinearLagrangeBasis {
  public:
    static constexpr int num_basis_functions = 2;

    struct Sample {
        Eigen::Matrix<double, num_basis_functions, 1> N;
        Eigen::Matrix<double, num_basis_functions, 1> dN_dxi;
    };

    // Evaluates the polynomials without clamping the parameter to the interval.
    static Sample evaluate(double t) {
        return {{1.0 - t, t}, {-1.0, 1.0}};
    }
};

} // namespace litepgo
