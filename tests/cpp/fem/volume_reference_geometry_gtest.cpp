#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/volume_reference_geometry.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using Basis = litepgo::HexQ1Basis;
using Reference = litepgo::VolumeReferenceGeometry<Basis>;
using Geometry = Reference::GeometryCoefficients;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

template <class B>
concept CanFormReference = requires { typename litepgo::VolumeReferenceGeometry<B>; };

static_assert(CanFormReference<Basis>);
static_assert(!CanFormReference<litepgo::LagrangeBasis>);
static_assert(Reference::num_basis_functions == 8);
static_assert(std::same_as<Geometry, Eigen::Matrix<double, 8, 3>>);
static_assert(std::same_as<decltype(std::declval<const Reference&>().evaluate(Vec3::Zero())), Reference::PointData>);

// Interior points, corners, a face, and an exterior point, independent of quadrature.
const std::array<Vec3, 7> parameters{{Vec3(0.2, 0.3, 0.4), Vec3(0.83, 0.17, 0.61), Vec3(0.5, 0.5, 0.5), Vec3(0, 0, 0),
                                      Vec3(1, 1, 1), Vec3(0, 0.3, 0.7), Vec3(-0.2, 1.2, 0.4)}};

Geometry corners() {
    Geometry X;
    X << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return X;
}

Geometry affine_geometry(const Mat3& J, const Vec3& origin) {
    return (corners() * J.transpose()).rowwise() + origin.transpose();
}

// Non-affine but globally invertible on the cube: det(J0) = 1 - alpha*beta*zeta^2.
constexpr double alpha = 0.4;
constexpr double beta = 0.2;

Vec3 warped_position(const Vec3& xi) {
    return {xi.x() + alpha * xi.y() * xi.z(), xi.y() + beta * xi.x() * xi.z(), xi.z()};
}

Vec3 warped_parameter(const Vec3& X) {
    const double determinant = 1.0 - alpha * beta * X.z() * X.z();
    return {(X.x() - alpha * X.z() * X.y()) / determinant, (X.y() - beta * X.z() * X.x()) / determinant, X.z()};
}

Geometry warped_geometry() {
    const auto nodes = corners();
    Geometry X;
    for (int i = 0; i < 8; ++i) {
        X.row(i) = warped_position(nodes.row(i).transpose()).transpose();
    }
    return X;
}

} // namespace

TEST(VolumeReferenceGeometry, UnitCubePreservesBasisAtArbitraryParameters) {
    const Reference reference(corners());
    for (const auto& xi : parameters) {
        const auto expected = Basis::evaluate(xi);
        const auto point = reference.evaluate(xi);
        EXPECT_TRUE(point.position.isApprox(xi, 1e-14));
        EXPECT_TRUE(point.jacobian.isApprox(Mat3::Identity(), 1e-14));
        EXPECT_TRUE(point.basis_values.isApprox(expected.values, 1e-14));
        EXPECT_TRUE(point.basis_gradients.isApprox(expected.gradients, 1e-14));
        EXPECT_TRUE(point.basis_gradients.colwise().sum().isZero(1e-14));
        EXPECT_NEAR(point.jacobian_determinant, 1.0, 1e-14);
    }
}

TEST(VolumeReferenceGeometry, ScalesGradientsForTranslatedBox) {
    const Vec3 lengths(2.0, 3.0, 4.0);
    const Mat3 expected_jacobian = lengths.asDiagonal();
    const Vec3 origin(1.0, -2.0, 3.0);
    const Reference reference(affine_geometry(expected_jacobian, origin));
    for (const auto& xi : parameters) {
        const auto expected = Basis::evaluate(xi);
        const auto point = reference.evaluate(xi);
        EXPECT_TRUE(point.position.isApprox(origin + lengths.asDiagonal() * xi, 1e-14));
        EXPECT_TRUE(point.jacobian.isApprox(expected_jacobian, 1e-14));
        for (int d = 0; d < 3; ++d) {
            EXPECT_TRUE(point.basis_gradients.col(d).isApprox(expected.gradients.col(d) / lengths[d], 1e-14));
        }
        EXPECT_NEAR(point.jacobian_determinant, 24.0, 1e-13);
    }
}

TEST(VolumeReferenceGeometry, MapsShearedHexAtArbitraryParameters) {
    Mat3 J;
    J << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Vec3 origin(1.2, -0.7, 2.1);
    const Reference reference(affine_geometry(J, origin));

    for (const auto& xi : parameters) {
        const auto point = reference.evaluate(xi);
        EXPECT_TRUE(point.position.isApprox(origin + J * xi, 1e-13));
        EXPECT_TRUE(point.jacobian.isApprox(J, 1e-13));
        EXPECT_NEAR(point.jacobian_determinant, J.determinant(), 1e-13);
    }
}

TEST(VolumeReferenceGeometry, ArbitraryPointGradientsMatchPhysicalFiniteDifferencesOnWarpedHex) {
    const Reference reference(warped_geometry());
    constexpr double h = 1e-6;
    for (const auto& xi : parameters) {
        const Vec3 position = warped_position(xi);
        const auto point = reference.evaluate(xi);
        EXPECT_TRUE(point.position.isApprox(position, 1e-14));
        EXPECT_NEAR(point.jacobian_determinant, 1.0 - alpha * beta * xi.z() * xi.z(), 1e-14);
        for (int d = 0; d < 3; ++d) {
            Vec3 plus = position;
            Vec3 minus = position;
            plus[d] += h;
            minus[d] -= h;
            const auto values_plus = Basis::evaluate(warped_parameter(plus)).values;
            const auto values_minus = Basis::evaluate(warped_parameter(minus)).values;
            const Reference::Values derivative = (values_plus - values_minus) / (2.0 * h);
            EXPECT_TRUE((point.basis_gradients.col(d) - derivative).isZero(2e-9));
        }
    }
}

TEST(VolumeReferenceGeometry, OwnsGeometryCoefficients) {
    Geometry X = corners();
    const Reference reference(X);
    X.setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto point = reference.evaluate(Vec3(0.2, 0.3, 0.4));
    EXPECT_NEAR(point.jacobian_determinant, 1.0, 1e-14);
    EXPECT_TRUE(point.basis_gradients.allFinite());
}

TEST(VolumeReferenceGeometry, RejectsNonFiniteGeometry) {
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        Geometry X = corners();
        X(4, 2) = value;
        EXPECT_THROW((Reference(X)), std::invalid_argument);
    }
}

TEST(VolumeReferenceGeometry, RejectsNonFiniteParameters) {
    const Reference reference(corners());
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        for (int d = 0; d < 3; ++d) {
            Vec3 xi(0.2, 0.3, 0.4);
            xi[d] = value;
            EXPECT_THROW(reference.evaluate(xi), std::invalid_argument);
        }
    }
}

TEST(VolumeReferenceGeometry, RejectsCollapsedAndInvertedGeometryWhenEvaluating) {
    const Vec3 xi(0.2, 0.3, 0.4);
    const Reference collapsed(Geometry::Zero());
    EXPECT_THROW(collapsed.evaluate(xi), std::invalid_argument);
    Geometry X = corners();
    X.col(2).setZero();
    const Reference flat(X);
    EXPECT_THROW(flat.evaluate(xi), std::invalid_argument);
    X = corners();
    X.col(0) *= -1.0;
    const Reference inverted(X);
    EXPECT_THROW(inverted.evaluate(xi), std::invalid_argument);
}

TEST(VolumeReferenceGeometry, ChecksJacobianAtTheRequestedPoint) {
    Geometry X = corners();
    // det(J0) = 1 - 1.5*xi: validity depends on the requested parameter point.
    for (int i = 0; i < 8; ++i) {
        X(i, 2) *= 1.0 - 1.5 * X(i, 0);
    }
    const Reference reference(X);
    EXPECT_NO_THROW(reference.evaluate(Vec3(0.2, 0.3, 0.4)));
    EXPECT_THROW(reference.evaluate(Vec3(0.8, 0.3, 0.4)), std::invalid_argument);
}

TEST(VolumeReferenceGeometry, AcceptsSmallPositiveJacobiansAndRejectsOverflowWhenEvaluating) {
    const Vec3 xi(0.2, 0.3, 0.4);
    const Reference small(1e-6 * corners());
    const auto point = small.evaluate(xi);
    EXPECT_GT(point.jacobian_determinant, 0.0);
    EXPECT_TRUE(point.basis_gradients.allFinite());
    const Reference enormous(1e200 * corners());
    EXPECT_THROW(enormous.evaluate(xi), std::invalid_argument);
}
