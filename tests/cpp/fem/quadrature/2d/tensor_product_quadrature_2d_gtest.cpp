#include "litepgo/fem/quadrature/2d/gauss_legendre_2d.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace litepgo::fem::test {

namespace {
static_assert(Quadrature2D<GaussLegendre2D<2>>);
static_assert(!Quadrature3D<GaussLegendre2D<2>>);
static_assert(GaussLegendre2D<2>::num_points == 4);
static_assert(std::same_as<decltype(GaussLegendre2D<2>::points()), const Eigen::Matrix<double, 4, 2>&>);

template <int Points> void check_monomials() {
    using Q = GaussLegendre2D<Points>;
    EXPECT_TRUE((Q::points().array() > 0).all());
    EXPECT_TRUE((Q::points().array() < 1).all());
    EXPECT_TRUE((Q::weights().array() > 0).all());
    EXPECT_NEAR(Q::weights().sum(), 1, 1e-15);
    for (int i = 0; i < 2 * Points; ++i)
        for (int j = 0; j < 2 * Points; ++j) {
            double integral = 0;
            for (int q = 0; q < Q::num_points; ++q)
                integral += Q::weights()[q] * std::pow(Q::points()(q, 0), i) * std::pow(Q::points()(q, 1), j);
            EXPECT_NEAR(integral, 1.0 / ((i + 1) * (j + 1)), 1e-15);
        }
}

struct UnequalWeightRule {
    static constexpr int num_points = 3;
    static const Eigen::Vector3d& points() {
        static const Eigen::Vector3d result(0, 0.25, 1);
        return result;
    }
    static const Eigen::Vector3d& weights() {
        static const Eigen::Vector3d result(0.2, 0.3, 0.5);
        return result;
    }
};

TEST(TensorProductQuadrature2D, IntegratesThroughGaussPolynomialDegreeInEachVariable) {
    check_monomials<2>();
    check_monomials<4>();
}

TEST(TensorProductQuadrature2D, OrdersPointsWithLastAxisFastestAndMultipliesUnequalWeights) {
    using Q = TensorProductQuadrature2D<UnequalWeightRule>;
    static_assert(Q::num_points == 9);
    Eigen::Matrix<double, 9, 2> expected;
    expected << 0, 0, 0, 0.25, 0, 1, 0.25, 0, 0.25, 0.25, 0.25, 1, 1, 0, 1, 0.25, 1, 1;
    EXPECT_EQ(Q::points(), expected);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(Q::weights()[3 * i + j],
                             UnequalWeightRule::weights()[i] * UnequalWeightRule::weights()[j]);
    EXPECT_NEAR(Q::weights().sum(), 1, 1e-15);
    EXPECT_NEAR(Q::weights().dot(Q::points().col(0)), 0.575, 1e-15);
    EXPECT_NEAR(Q::weights().dot(Q::points().col(1)), 0.575, 1e-15);
}
} // namespace

} // namespace litepgo::fem::test
