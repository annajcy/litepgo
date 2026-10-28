#pragma once

#include "litepgo/fem/basis/basis_sample.hpp"

namespace litepgo {

// Linear nodal basis on [0, 1], ordered by the nodes 0, 1 (libpgo convention).
class LinearLagrangeBasis {
  public:
    static constexpr int num_basis_functions = 2;

    using Sample = BasisSample<num_basis_functions, 1>;

    // Evaluates the polynomials without clamping the parameter to the interval.
    static Sample evaluate(double t) {
        return {{1.0 - t, t}, {-1.0, 1.0}, {0.0, 0.0}};
    }
};

} // namespace litepgo
