#pragma once

#include "litepgo/fem/basis/basis.hpp"
#include <stdexcept>

namespace litepgo::fem {

// Linear nodal basis on [0, 1], ordered by the nodes 0, 1 (libpgo convention).
class LinearLagrangeBasis {
  public:
    static constexpr int num_basis_functions = 2;

    static CoefficientInfo<1> coefficient_info(int i) {
        if (i < 0 || i >= num_basis_functions)
            throw std::out_of_range("LinearLagrangeBasis::coefficient_info: index out of range");
        return {static_cast<double>(i), {0}};
    }

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

} // namespace litepgo::fem
