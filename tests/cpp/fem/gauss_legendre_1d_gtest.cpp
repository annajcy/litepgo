#include "litepgo/fem/gauss_legendre_1d.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>

namespace {

using Rule = litepgo::GaussLegendre1D<2>;
static_assert(Rule::num_points == 2);
static_assert(std::same_as<decltype(Rule::points()), const Eigen::Vector2d&>);
static_assert(std::same_as<decltype(Rule::weights()), const Eigen::Vector2d&>);

} // namespace

TEST(GaussLegendre1D, HasOrderedInteriorPointsAndNormalizedWeights) {
    const auto& points = Rule::points();
    const auto& weights = Rule::weights();
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
        for (int q = 0; q < Rule::num_points; ++q) {
            integral += Rule::weights()[q] * std::pow(Rule::points()[q], degree);
        }
        EXPECT_NEAR(integral, 1.0 / (degree + 1), 1e-15);
    }
}
