#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"

#include <gtest/gtest.h>

using litepgo::LinearLagrangeBasis;

static_assert(LinearLagrangeBasis::num_basis_functions == 2);

TEST(LinearLagrangeBasis, InterpolatesEndpointValues) {
    const auto left_N = LinearLagrangeBasis::eval_N(0.0);
    const auto right_N = LinearLagrangeBasis::eval_N(1.0);
    EXPECT_DOUBLE_EQ(left_N[0], 1.0);
    EXPECT_DOUBLE_EQ(left_N[1], 0.0);
    EXPECT_DOUBLE_EQ(right_N[0], 0.0);
    EXPECT_DOUBLE_EQ(right_N[1], 1.0);
}

TEST(LinearLagrangeBasis, ReproducesLinearFunctionsAndDerivatives) {
    // Exterior points also check that evaluation does not clamp the parameter.
    for (double t : {-0.5, 0.0, 0.13, 0.5, 0.61, 1.0, 1.5}) {
        const auto N = LinearLagrangeBasis::eval_N(t);
        const auto dN_dxi = LinearLagrangeBasis::eval_dN_dxi(t);
        const auto d2N_dxi2 = LinearLagrangeBasis::eval_d2N_dxi2(t);
        EXPECT_NEAR(N[0] + N[1], 1.0, 1e-15);
        EXPECT_DOUBLE_EQ(dN_dxi[0] + dN_dxi[1], 0.0);
        EXPECT_TRUE(d2N_dxi2.isZero());

        // f(t) = 2 + 3t, with endpoint values 2 and 5.
        EXPECT_NEAR(2.0 * N[0] + 5.0 * N[1], 2.0 + 3.0 * t, 2e-15);
        EXPECT_DOUBLE_EQ(2.0 * dN_dxi[0] + 5.0 * dN_dxi[1], 3.0);

        constexpr double h = 1e-6;
        const auto plus_N = LinearLagrangeBasis::eval_N(t + h);
        const auto plus_dN_dxi = LinearLagrangeBasis::eval_dN_dxi(t + h);
        const auto minus_N = LinearLagrangeBasis::eval_N(t - h);
        const auto minus_dN_dxi = LinearLagrangeBasis::eval_dN_dxi(t - h);
        for (int a = 0; a < LinearLagrangeBasis::num_basis_functions; ++a) {
            EXPECT_NEAR((plus_N[a] - minus_N[a]) / (2.0 * h), dN_dxi[a], 2e-10);
            EXPECT_DOUBLE_EQ((plus_dN_dxi[a] - minus_dN_dxi[a]) / (2.0 * h), d2N_dxi2[a]);
        }
    }
}
