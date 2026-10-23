#include "litepgo/fem/gauss_legendre_3d.h"
#include "litepgo/fem/hex_q1_basis.h"
#include "litepgo/fem/volume_quadrature_cache.h"

#include <gtest/gtest.h>

#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using Basis = litepgo::HexQ1Basis;
using Quadrature = litepgo::GaussLegendre3D<2>;
using Reference = litepgo::VolumeReferenceGeometry<Basis>;
using Cache = litepgo::VolumeQuadratureCache<Basis, Quadrature>;
using Geometry = Reference::GeometryCoefficients;

template <class B, class Q>
concept CanFormCache = requires { typename litepgo::VolumeQuadratureCache<B, Q>; };

static_assert(CanFormCache<Basis, Quadrature>);
static_assert(!CanFormCache<litepgo::LagrangeBasis, Quadrature>);
static_assert(!CanFormCache<Basis, litepgo::GaussLegendre1D<2>>);
static_assert(Cache::num_quadrature_points == 8);
static_assert(std::same_as<Cache::PointData, Reference::PointData>);
static_assert(std::same_as<decltype(std::declval<const Cache&>().sample(0)), const Cache::Sample&>);

Geometry corners() {
    Geometry X;
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
    const Reference reference(corners());
    const Cache cache(reference);
    double volume = 0.0;
    for (int q = 0; q < Cache::num_quadrature_points; ++q) {
        const auto expected = reference.evaluate(Quadrature::points().row(q).transpose());
        const auto& sample = cache.sample(q);
        EXPECT_EQ(sample.point.values, expected.values);
        EXPECT_EQ(sample.point.gradients, expected.gradients);
        EXPECT_EQ(sample.point.jacobian_determinant, expected.jacobian_determinant);
        EXPECT_EQ(sample.point.B(), expected.B());
        EXPECT_EQ(sample.point.BF(), expected.BF());
        EXPECT_NEAR(sample.volume_weight, 0.125, 1e-15);
        volume += sample.volume_weight;
    }
    EXPECT_NEAR(volume, 1.0, 1e-14);
}

TEST(VolumeQuadratureCache, IntegratesTranslatedBoxVolume) {
    Geometry X = corners() * Eigen::Vector3d(2.0, 3.0, 4.0).asDiagonal();
    X.rowwise() += Eigen::RowVector3d(1.0, -2.0, 3.0);
    const Reference reference(X);
    const Cache cache(reference);
    double volume = 0.0;
    for (int q = 0; q < Cache::num_quadrature_points; ++q) {
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
    Geometry X;
    for (int i = 0; i < 8; ++i) {
        const double x = nodes(i, 0);
        const double y = nodes(i, 1);
        const double z = nodes(i, 2);
        X.row(i) << x + alpha * y * z, y + beta * x * z, z;
    }
    const Reference reference(X);
    const Cache cache(reference);
    double volume = 0.0;
    for (int q = 0; q < Cache::num_quadrature_points; ++q) {
        const double z = Quadrature::points()(q, 2);
        const double determinant = 1.0 - alpha * beta * z * z;
        const auto& sample = cache.sample(q);
        EXPECT_NEAR(sample.point.jacobian_determinant, determinant, 1e-14);
        EXPECT_NEAR(sample.volume_weight, determinant / 8.0, 1e-14);
        volume += sample.volume_weight;
    }
    EXPECT_NEAR(volume, 1.0 - alpha * beta / 3.0, 1e-14);
}

TEST(VolumeQuadratureCache, OwnsSamplesAfterReferenceGeometryIsDestroyed) {
    const Cache cache{Reference(corners())};
    const auto& sample = cache.sample(0);
    EXPECT_NEAR(sample.point.jacobian_determinant, 1.0, 1e-14);
    EXPECT_NEAR(sample.volume_weight, 0.125, 1e-15);
    EXPECT_TRUE(sample.point.B().allFinite());
    EXPECT_TRUE(sample.point.BF().allFinite());
}

TEST(VolumeQuadratureCache, ChecksQuadraturePointIndices) {
    const Reference reference(corners());
    const Cache cache(reference);
    EXPECT_NO_THROW(cache.sample(0));
    EXPECT_NO_THROW(cache.sample(7));
    EXPECT_THROW(cache.sample(-1), std::out_of_range);
    EXPECT_THROW(cache.sample(8), std::out_of_range);
}

TEST(VolumeQuadratureCache, PropagatesInversionAtLaterQuadraturePoints) {
    Geometry X = corners();
    // det(J0) is positive at the first four points and negative at the last four.
    for (int i = 0; i < 8; ++i) {
        X(i, 2) *= 1.0 - 1.5 * X(i, 0);
    }
    const Reference reference(X);
    EXPECT_NO_THROW(reference.evaluate(Quadrature::points().row(0).transpose()));
    EXPECT_THROW((Cache(reference)), std::invalid_argument);
}

TEST(VolumeQuadratureCache, SupportsOtherQuadratureRules) {
    using OtherCache = litepgo::VolumeQuadratureCache<Basis, SinglePointRule>;
    static_assert(OtherCache::num_quadrature_points == 1);
    const Reference reference(corners());
    const OtherCache cache(reference);
    const auto expected = reference.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4));
    EXPECT_EQ(cache.sample(0).point.values, expected.values);
    EXPECT_EQ(cache.sample(0).point.gradients, expected.gradients);
    EXPECT_NEAR(cache.sample(0).volume_weight, 0.75, 1e-14);
}

TEST(VolumeQuadratureCache, RejectsNonFiniteAndOverflowingVolumeWeights) {
    using NonFiniteCache = litepgo::VolumeQuadratureCache<Basis, NonFiniteWeightRule>;
    using OverflowingCache = litepgo::VolumeQuadratureCache<Basis, OverflowingWeightRule>;
    const Reference reference(2.0 * corners());
    EXPECT_THROW((NonFiniteCache(reference)), std::invalid_argument);
    EXPECT_THROW((OverflowingCache(reference)), std::invalid_argument);
}
