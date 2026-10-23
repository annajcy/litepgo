#pragma once

#include <Eigen/Core>

namespace litepgo {

// Linear nodal basis on [-1, 1], ordered by the nodes -1, +1.
class LagrangeBasis {
  public:
    static constexpr int num_basis_functions = 2;

    struct Sample {
        Eigen::Matrix<double, num_basis_functions, 1> values;
        Eigen::Matrix<double, num_basis_functions, 1> gradients;
    };

    // Evaluates the polynomials without clamping the parameter to the interval.
    static Sample evaluate(double t) {
        return {{0.5 * (1.0 - t), 0.5 * (1.0 + t)}, {-0.5, 0.5}};
    }
};

} // namespace litepgo
