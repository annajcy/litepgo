#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>

namespace litepgo::fem::test {

namespace {

static_assert(litepgo::fem::GaussLegendre1D<2>::num_points == 2);
static_assert(std::same_as<decltype(litepgo::fem::GaussLegendre1D<2>::points()), const Eigen::Vector2d&>);
static_assert(std::same_as<decltype(litepgo::fem::GaussLegendre1D<2>::weights()), const Eigen::Vector2d&>);
static_assert(litepgo::fem::Quadrature1D<litepgo::fem::GaussLegendre1D<4>>);

} // namespace

TEST(GaussLegendre1D, HasOrderedInteriorPointsAndNormalizedWeights) {
    const auto& points = litepgo::fem::GaussLegendre1D<2>::points();
    const auto& weights = litepgo::fem::GaussLegendre1D<2>::weights();
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
        for (int q = 0; q < litepgo::fem::GaussLegendre1D<2>::num_points; ++q) {
            integral += litepgo::fem::GaussLegendre1D<2>::weights()[q] *
                        std::pow(litepgo::fem::GaussLegendre1D<2>::points()[q], degree);
        }
        EXPECT_NEAR(integral, 1.0 / (degree + 1), 1e-15);
    }
}

TEST(GaussLegendre1D, FourPointsIntegrateMonomialsThroughDegreeSeven) {
    const auto& points = litepgo::fem::GaussLegendre1D<4>::points();
    const auto& weights = litepgo::fem::GaussLegendre1D<4>::weights();
    EXPECT_GT(points[0], 0);
    EXPECT_LT(points[3], 1);
    EXPECT_GT(weights.minCoeff(), 0);
    for (int q = 0; q < 4; ++q) {
        EXPECT_NEAR(points[q] + points[3 - q], 1, 1e-15);
        EXPECT_DOUBLE_EQ(weights[q], weights[3 - q]);
        if (q > 0)
            EXPECT_GT(points[q], points[q - 1]);
    }
    for (int degree = 0; degree <= 7; ++degree) {
        SCOPED_TRACE(degree);
        EXPECT_NEAR(weights.dot(points.array().pow(degree).matrix()), 1.0 / (degree + 1), 1e-15);
    }
}

} // namespace litepgo::fem::test
