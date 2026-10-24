#include "litepgo/fem/gauss_legendre_1d.hpp"
#include "litepgo/fem/gauss_legendre_3d.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/tensor_product_quadrature.hpp"
#include "litepgo/fem/volume_quadrature_cache.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

template <class B, class Q>
concept CanFormCache = requires { typename litepgo::VolumeQuadratureCache<B, Q>; };

static_assert(CanFormCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>);
static_assert(!CanFormCache<litepgo::LagrangeBasis, litepgo::GaussLegendre3D<2>>);
static_assert(!CanFormCache<litepgo::HexQ1Basis, litepgo::GaussLegendre1D<2>>);
static_assert(litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>::num_quadrature_points ==
              8);
static_assert(
    std::same_as<
        decltype(std::declval<const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>&>()
                     .sample(0)
                     .point),
        litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
static_assert(
    std::same_as<
        decltype(std::declval<const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>&>()
                     .sample(0)),
        const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>::Sample&>);

Eigen::Matrix<double, 8, 3> corners() {
    Eigen::Matrix<double, 8, 3> X;
    X << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return X;
}

struct SinglePointRule {
    static constexpr int num_points = 1;

    static const Eigen::Matrix<double, 1, 3>& points() {
        static const Eigen::Matrix<double, 1, 3> result{0.2, 0.3, 0.4};
        return result;
    }

    static const Eigen::Matrix<double, 1, 1>& weights() {
        static const Eigen::Matrix<double, 1, 1> result{0.75};
        return result;
    }
};

struct NonFiniteWeightRule : SinglePointRule {
    static const Eigen::Matrix<double, 1, 1>& weights() {
        static const Eigen::Matrix<double, 1, 1> result{std::numeric_limits<double>::quiet_NaN()};
        return result;
    }
};

struct OverflowingWeightRule : SinglePointRule {
    static const Eigen::Matrix<double, 1, 1>& weights() {
        static const Eigen::Matrix<double, 1, 1> result{std::numeric_limits<double>::max()};
        return result;
    }
};

} // namespace

TEST(VolumeQuadratureCache, MatchesDirectPointEvaluationAndIntegratesUnitVolume) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(corners());
    const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>> cache(reference);
    double volume = 0.0;
    for (int q = 0;
         q < litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>::num_quadrature_points;
         ++q) {
        const auto expected = reference.evaluate(litepgo::GaussLegendre3D<2>::points().row(q).transpose());
        const auto& sample = cache.sample(q);
        EXPECT_EQ(sample.point.position, expected.position);
        EXPECT_EQ(sample.point.jacobian, expected.jacobian);
        EXPECT_EQ(sample.point.basis_values, expected.basis_values);
        EXPECT_EQ(sample.point.basis_gradients, expected.basis_gradients);
        EXPECT_EQ(sample.point.jacobian_determinant, expected.jacobian_determinant);
        EXPECT_NEAR(sample.volume_weight, 0.125, 1e-15);
        volume += sample.volume_weight;
    }
    EXPECT_NEAR(volume, 1.0, 1e-14);
}

TEST(VolumeQuadratureCache, IntegratesTranslatedBoxVolume) {
    Eigen::Matrix<double, 8, 3> X = corners() * Eigen::Vector3d(2.0, 3.0, 4.0).asDiagonal();
    X.rowwise() += Eigen::RowVector3d(1.0, -2.0, 3.0);
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(X);
    const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>> cache(reference);
    double volume = 0.0;
    for (int q = 0;
         q < litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>::num_quadrature_points;
         ++q) {
        const auto& sample = cache.sample(q);
        EXPECT_NEAR(sample.point.jacobian_determinant, 24.0, 1e-13);
        EXPECT_NEAR(sample.volume_weight, 3.0, 1e-14);
        volume += sample.volume_weight;
    }
    EXPECT_NEAR(volume, 24.0, 1e-13);
}

TEST(VolumeQuadratureCache, WarpedHexUsesPointwiseJacobiansAndIntegratesVolume) {
    constexpr double alpha = 0.4;
    constexpr double beta = 0.2;
    const auto nodes = corners();
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i) {
        const double x = nodes(i, 0);
        const double y = nodes(i, 1);
        const double z = nodes(i, 2);
        X.row(i) << x + alpha * y * z, y + beta * x * z, z;
    }
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(X);
    const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>> cache(reference);
    double volume = 0.0;
    for (int q = 0;
         q < litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>::num_quadrature_points;
         ++q) {
        const double z = litepgo::GaussLegendre3D<2>::points()(q, 2);
        const double determinant = 1.0 - alpha * beta * z * z;
        const auto& sample = cache.sample(q);
        EXPECT_NEAR(sample.point.jacobian_determinant, determinant, 1e-14);
        EXPECT_NEAR(sample.volume_weight, determinant / 8.0, 1e-14);
        volume += sample.volume_weight;
    }
    EXPECT_NEAR(volume, 1.0 - alpha * beta / 3.0, 1e-14);
}

TEST(VolumeQuadratureCache, OwnsSamplesAfterReferenceGeometryIsDestroyed) {
    const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>> cache{
        litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>(corners())};
    const auto& sample = cache.sample(0);
    EXPECT_NEAR(sample.point.jacobian_determinant, 1.0, 1e-14);
    EXPECT_NEAR(sample.volume_weight, 0.125, 1e-15);
    EXPECT_TRUE(sample.point.position.allFinite());
    EXPECT_TRUE(sample.point.jacobian.allFinite());
}

TEST(VolumeQuadratureCache, ChecksQuadraturePointIndices) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(corners());
    const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>> cache(reference);
    EXPECT_NO_THROW(cache.sample(0));
    EXPECT_NO_THROW(cache.sample(7));
    EXPECT_THROW(cache.sample(-1), std::out_of_range);
    EXPECT_THROW(cache.sample(8), std::out_of_range);
}

TEST(VolumeQuadratureCache, PropagatesInversionAtLaterQuadraturePoints) {
    Eigen::Matrix<double, 8, 3> X = corners();
    // det(J0) is positive at the first four points and negative at the last four.
    for (int i = 0; i < 8; ++i) {
        X(i, 2) *= 1.0 - 1.5 * X(i, 0);
    }
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(X);
    EXPECT_NO_THROW(reference.evaluate(litepgo::GaussLegendre3D<2>::points().row(0).transpose()));
    EXPECT_THROW((litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, litepgo::GaussLegendre3D<2>>(reference)),
                 std::invalid_argument);
}

TEST(VolumeQuadratureCache, SupportsOtherQuadratureRules) {

    static_assert(litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, SinglePointRule>::num_quadrature_points == 1);
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(corners());
    const litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, SinglePointRule> cache(reference);
    const auto expected = reference.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4));
    EXPECT_EQ(cache.sample(0).point.basis_values, expected.basis_values);
    EXPECT_EQ(cache.sample(0).point.basis_gradients, expected.basis_gradients);
    EXPECT_NEAR(cache.sample(0).volume_weight, 0.75, 1e-14);
}

TEST(VolumeQuadratureCache, RejectsNonFiniteAndOverflowingVolumeWeights) {

    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(2.0 * corners());
    EXPECT_THROW((litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, NonFiniteWeightRule>(reference)),
                 std::invalid_argument);
    EXPECT_THROW((litepgo::VolumeQuadratureCache<litepgo::HexQ1Basis, OverflowingWeightRule>(reference)),
                 std::invalid_argument);
}
