#include "litepgo/fem/lagrange_basis.h"

#include <gtest/gtest.h>

using litepgo::LagrangeBasis;

static_assert(LagrangeBasis::num_basis_functions == 2);

TEST(LagrangeBasis, InterpolatesEndpointValues) {
    const auto left = LagrangeBasis::evaluate(0.0);
    const auto right = LagrangeBasis::evaluate(1.0);
    EXPECT_DOUBLE_EQ(left.values[0], 1.0);
    EXPECT_DOUBLE_EQ(left.values[1], 0.0);
    EXPECT_DOUBLE_EQ(right.values[0], 0.0);
    EXPECT_DOUBLE_EQ(right.values[1], 1.0);
}

TEST(LagrangeBasis, ReproducesLinearFunctionsAndDerivatives) {
    // Exterior points also check that evaluation does not clamp the parameter.
    for (double t : {-0.5, 0.0, 0.13, 0.5, 0.61, 1.0, 1.5}) {
        const auto sample = LagrangeBasis::evaluate(t);
        EXPECT_NEAR(sample.values[0] + sample.values[1], 1.0, 1e-15);
        EXPECT_DOUBLE_EQ(sample.gradients[0] + sample.gradients[1], 0.0);

        // f(t) = 2 + 3t, with endpoint values 2 and 5.
        EXPECT_NEAR(2.0 * sample.values[0] + 5.0 * sample.values[1], 2.0 + 3.0 * t, 2e-15);
        EXPECT_DOUBLE_EQ(2.0 * sample.gradients[0] + 5.0 * sample.gradients[1], 3.0);

        constexpr double h = 1e-6;
        const auto plus = LagrangeBasis::evaluate(t + h);
        const auto minus = LagrangeBasis::evaluate(t - h);
        for (int a = 0; a < LagrangeBasis::num_basis_functions; ++a) {
            EXPECT_NEAR((plus.values[a] - minus.values[a]) / (2.0 * h), sample.gradients[a], 2e-10);
        }
    }
}
