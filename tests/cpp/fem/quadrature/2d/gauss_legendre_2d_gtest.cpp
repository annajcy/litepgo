#include "litepgo/fem/quadrature/2d/gauss_legendre_2d.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace litepgo::fem::test {
namespace {

static_assert(Quadrature2D<GaussLegendre2D<2>>);
static_assert(Quadrature2D<GaussLegendre2D<4>>);
static_assert(GaussLegendre2D<2>::num_points == 4);
static_assert(GaussLegendre2D<4>::num_points == 16);

template <int N> void check_order_and_weights() {
    using Q = GaussLegendre2D<N>;
    const auto& points = Q::points();
    const auto& weights = Q::weights();
    const auto& t = GaussLegendre1D<N>::points();
    const auto& w = GaussLegendre1D<N>::weights();
    EXPECT_EQ(&points, &Q::points());
    EXPECT_EQ(&weights, &Q::weights());
    EXPECT_TRUE((points.array() > 0).all());
    EXPECT_TRUE((points.array() < 1).all());
    EXPECT_TRUE((weights.array() > 0).all());
    EXPECT_NEAR(weights.sum(), 1, 1e-15);
    for (int q = 0; q < Q::num_points; ++q) {
        int index = q;
        double expected_weight = 1;
        for (int axis = 1; axis >= 0; --axis) {
            EXPECT_DOUBLE_EQ(points(q, axis), t[index % N]);
            expected_weight *= w[index % N];
            index /= N;
        }
        EXPECT_NEAR(weights[q], expected_weight, 1e-16);
    }
}

template <int N> void check_monomials() {
    using Q = GaussLegendre2D<N>;
    for (int i = 0; i < 2 * N; ++i)
        for (int j = 0; j < 2 * N; ++j) {
            double integral = 0;
            for (int q = 0; q < Q::num_points; ++q)
                integral += Q::weights()[q] * std::pow(Q::points()(q, 0), i) * std::pow(Q::points()(q, 1), j);
            EXPECT_NEAR(integral, 1.0 / ((i + 1) * (j + 1)), 1e-15);
        }
}

TEST(GaussLegendre2D, OrdersPointsWithLastAxisFastestAndMultipliesWeights) {
    check_order_and_weights<2>();
    check_order_and_weights<4>(); // Four-point weights are unequal.
}

TEST(GaussLegendre2D, IntegratesThroughGaussPolynomialDegreeInEachVariable) {
    check_monomials<2>();
    check_monomials<4>();
}

} // namespace
} // namespace litepgo::fem::test
