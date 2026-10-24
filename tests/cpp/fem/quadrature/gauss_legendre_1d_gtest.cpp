#include "litepgo/fem/quadrature/gauss_legendre_1d.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>

namespace {

static_assert(litepgo::GaussLegendre1D<2>::num_points == 2);
static_assert(std::same_as<decltype(litepgo::GaussLegendre1D<2>::points()), const Eigen::Vector2d&>);
static_assert(std::same_as<decltype(litepgo::GaussLegendre1D<2>::weights()), const Eigen::Vector2d&>);

} // namespace

TEST(GaussLegendre1D, HasOrderedInteriorPointsAndNormalizedWeights) {
    const auto& points = litepgo::GaussLegendre1D<2>::points();
    const auto& weights = litepgo::GaussLegendre1D<2>::weights();
    EXPECT_NEAR(points[0], 0.21132486540518713, 1e-15);
    EXPECT_NEAR(points[1], 0.7886751345948129, 1e-15);
    EXPECT_GT(points[0], 0.0);
    EXPECT_LT(points[0], points[1]);
    EXPECT_LT(points[1], 1.0);
    EXPECT_DOUBLE_EQ(weights[0], 0.5);
    EXPECT_DOUBLE_EQ(weights[1], 0.5);
    EXPECT_DOUBLE_EQ(weights.sum(), 1.0);
}

TEST(GaussLegendre1D, IntegratesMonomialsThroughDegreeThree) {
    for (int degree = 0; degree <= 3; ++degree) {
        SCOPED_TRACE(degree);
        double integral = 0.0;
        for (int q = 0; q < litepgo::GaussLegendre1D<2>::num_points; ++q) {
            integral +=
                litepgo::GaussLegendre1D<2>::weights()[q] * std::pow(litepgo::GaussLegendre1D<2>::points()[q], degree);
        }
        EXPECT_NEAR(integral, 1.0 / (degree + 1), 1e-15);
    }
}
