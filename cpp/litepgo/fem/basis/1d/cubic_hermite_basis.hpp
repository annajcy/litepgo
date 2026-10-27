#pragma once

#include <Eigen/Core>

namespace litepgo {

// Cubic Hermite basis on [0,1], ordered as u(0), u'(0), u(1), u'(1).
// Derivative coefficients are with respect to the reference parameter t.
class CubicHermiteBasis {
  public:
    static constexpr int num_basis_functions = 4;

    struct Sample {
        Eigen::Vector4d N;
        Eigen::Vector4d dN_dxi;
    };

    // Evaluates the polynomials without clamping the parameter to the interval.
    static Sample evaluate(double t) {
        const double t2 = t * t, t3 = t2 * t;
        return {{2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2},
                {6 * t2 - 6 * t, 3 * t2 - 4 * t + 1, -6 * t2 + 6 * t, 3 * t2 - 2 * t}};
    }
};

} // namespace litepgo
