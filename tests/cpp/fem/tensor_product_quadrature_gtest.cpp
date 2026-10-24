#include "litepgo/fem/gauss_legendre_1d.hpp"
#include "litepgo/fem/gauss_legendre_3d.hpp"
#include "litepgo/fem/tensor_product_quadrature.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>

namespace {

static_assert(std::same_as<litepgo::GaussLegendre3D<2>, litepgo::GaussLegendre3D<2>>);
static_assert(litepgo::GaussLegendre3D<2>::num_points == 8);
static_assert(std::same_as<decltype(litepgo::GaussLegendre3D<2>::points()), const Eigen::Matrix<double, 8, 3>&>);
static_assert(std::same_as<decltype(litepgo::GaussLegendre3D<2>::weights()), const Eigen::Matrix<double, 8, 1>&>);

// Unequal weights and three points exercise the generic rule independently of
// the two-point Gauss rule's uniform weights.
struct UnequalWeightRule {
    static constexpr int num_points = 3;

    static const Eigen::Vector3d& points() {
        static const Eigen::Vector3d result{0.0, 0.25, 1.0};
        return result;
    }

    static const Eigen::Vector3d& weights() {
        static const Eigen::Vector3d result{0.2, 0.3, 0.5};
        return result;
    }
};

} // namespace

TEST(TensorProductQuadrature, OrdersPointsWithLastAxisFastest) {
    constexpr double a = 0.21132486540518713;
    constexpr double b = 0.7886751345948129;
    Eigen::Matrix<double, 8, 3> expected;
    expected << a, a, a, a, a, b, a, b, a, a, b, b, b, a, a, b, a, b, b, b, a, b, b, b;
    EXPECT_TRUE(litepgo::GaussLegendre3D<2>::points().isApprox(expected, 1e-15));
    EXPECT_TRUE((litepgo::GaussLegendre3D<2>::points().array() > 0.0).all());
    EXPECT_TRUE((litepgo::GaussLegendre3D<2>::points().array() < 1.0).all());
    for (double weight : litepgo::GaussLegendre3D<2>::weights()) {
        EXPECT_DOUBLE_EQ(weight, 0.125);
    }
    EXPECT_DOUBLE_EQ(litepgo::GaussLegendre3D<2>::weights().sum(), 1.0);
}

TEST(TensorProductQuadrature, IntegratesMonomialsThroughDegreeThreeInEachVariable) {
    for (int i = 0; i <= 3; ++i) {
        for (int j = 0; j <= 3; ++j) {
            for (int k = 0; k <= 3; ++k) {
                SCOPED_TRACE(::testing::Message() << "degrees: " << i << ", " << j << ", " << k);
                double integral = 0.0;
                for (int q = 0; q < litepgo::GaussLegendre3D<2>::num_points; ++q) {
                    const auto point = litepgo::GaussLegendre3D<2>::points().row(q);
                    integral += litepgo::GaussLegendre3D<2>::weights()[q] * std::pow(point[0], i) *
                                std::pow(point[1], j) * std::pow(point[2], k);
                }
                EXPECT_NEAR(integral, 1.0 / ((i + 1) * (j + 1) * (k + 1)), 1e-15);
            }
        }
    }
}

TEST(TensorProductQuadrature, SupportsOtherPointCountsAndUnequalWeights) {

    static_assert(litepgo::TensorProductQuadrature3D<UnequalWeightRule>::num_points == 27);
    const auto& points = litepgo::TensorProductQuadrature3D<UnequalWeightRule>::points();
    const auto& weights = litepgo::TensorProductQuadrature3D<UnequalWeightRule>::weights();
    EXPECT_TRUE(points.row(0).isZero());
    EXPECT_TRUE(points.row(1).transpose().isApprox(Eigen::Vector3d(0.0, 0.0, 0.25)));
    EXPECT_TRUE(points.row(3).transpose().isApprox(Eigen::Vector3d(0.0, 0.25, 0.0)));
    EXPECT_TRUE(points.row(9).transpose().isApprox(Eigen::Vector3d(0.25, 0.0, 0.0)));
    EXPECT_TRUE(points.row(26).isOnes());
    EXPECT_NEAR(weights[0], 0.008, 1e-15);
    EXPECT_NEAR(weights[1], 0.012, 1e-15);
    EXPECT_NEAR(weights[5], 0.03, 1e-15);
    EXPECT_NEAR(weights[26], 0.125, 1e-15);
    EXPECT_TRUE((weights.array() > 0.0).all());
    EXPECT_NEAR(weights.sum(), 1.0, 1e-15);
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(weights.dot(points.col(d)), 0.575, 1e-15);
    }
}
