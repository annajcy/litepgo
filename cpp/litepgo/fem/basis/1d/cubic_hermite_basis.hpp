#pragma once

#include <Eigen/Core>

namespace litepgo {

// Cubic Hermite basis on [0,1], ordered as u(0), u'(0), u(1), u'(1).
// Derivative coefficients are with respect to the reference parameter t.
class CubicHermiteBasis {
  public:
    static constexpr int num_basis_functions = 4;

    // Evaluates the polynomials without clamping the parameter to the interval.
    static Eigen::Vector4d evaluate_N(double t) {
        const double t2 = t * t, t3 = t2 * t;
        return {2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2};
    }
    static Eigen::Vector4d evaluate_dN_dxi(double t) {
        const double t2 = t * t;
        return {6 * t2 - 6 * t, 3 * t2 - 4 * t + 1, -6 * t2 + 6 * t, 3 * t2 - 2 * t};
    }
    static Eigen::Vector4d evaluate_d2N_dxi2(double t) {
        return {12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2};
    }
};

} // namespace litepgo
