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

template <class B>
concept CanFormReference = requires { typename litepgo::VolumeReferenceGeometry<B>; };

static_assert(CanFormReference<litepgo::HexQ1Basis>);
static_assert(!CanFormReference<litepgo::LagrangeBasis>);
static_assert(litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::num_basis_functions == 8);
static_assert(std::same_as<Eigen::Matrix<double, 8, 3>, Eigen::Matrix<double, 8, 3>>);
static_assert(std::same_as<decltype(std::declval<const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>&>()
                                        .evaluate(Eigen::Vector3d::Zero())),
                           litepgo::VolumePointData>);

// Interior points, corners, a face, and an exterior point, independent of quadrature.
const std::array<Eigen::Vector3d, 7> parameters{{Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(0.83, 0.17, 0.61),
                                                 Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(0, 0, 0),
                                                 Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0, 0.3, 0.7),
                                                 Eigen::Vector3d(-0.2, 1.2, 0.4)}};

Eigen::Matrix<double, 8, 3> corners() {
    Eigen::Matrix<double, 8, 3> X;
    X << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return X;
}

Eigen::Matrix<double, 8, 3> affine_geometry(const Eigen::Matrix3d& J, const Eigen::Vector3d& origin) {
    return (corners() * J.transpose()).rowwise() + origin.transpose();
}

// Non-affine but globally invertible on the cube: det(J0) = 1 - alpha*beta*zeta^2.
constexpr double alpha = 0.4;
constexpr double beta = 0.2;

Eigen::Vector3d warped_position(const Eigen::Vector3d& xi) {
    return {xi.x() + alpha * xi.y() * xi.z(), xi.y() + beta * xi.x() * xi.z(), xi.z()};
}

Eigen::Vector3d warped_parameter(const Eigen::Vector3d& X) {
    const double determinant = 1.0 - alpha * beta * X.z() * X.z();
    return {(X.x() - alpha * X.z() * X.y()) / determinant, (X.y() - beta * X.z() * X.x()) / determinant, X.z()};
}

Eigen::Matrix<double, 8, 3> warped_geometry() {
    const auto nodes = corners();
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i) {
        X.row(i) = warped_position(nodes.row(i).transpose()).transpose();
    }
    return X;
}

} // namespace

TEST(VolumeReferenceGeometry, UnitCubePreservesBasisAtArbitraryParameters) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(corners());
    for (const auto& xi : parameters) {
        const auto expected = litepgo::HexQ1Basis::evaluate(xi);
        const auto point = reference.evaluate(xi);
        EXPECT_TRUE(point.position.isApprox(xi, 1e-14));
        EXPECT_TRUE(point.jacobian.isApprox(Eigen::Matrix3d::Identity(), 1e-14));
        EXPECT_TRUE(point.basis_values.isApprox(expected.values, 1e-14));
        EXPECT_TRUE(point.basis_gradients.isApprox(expected.gradients, 1e-14));
        EXPECT_TRUE(point.basis_gradients.colwise().sum().isZero(1e-14));
        EXPECT_NEAR(point.jacobian_determinant, 1.0, 1e-14);
    }
}

TEST(VolumeReferenceGeometry, ScalesGradientsForTranslatedBox) {
    const Eigen::Vector3d lengths(2.0, 3.0, 4.0);
    const Eigen::Matrix3d expected_jacobian = lengths.asDiagonal();
    const Eigen::Vector3d origin(1.0, -2.0, 3.0);
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(affine_geometry(expected_jacobian, origin));
    for (const auto& xi : parameters) {
        const auto expected = litepgo::HexQ1Basis::evaluate(xi);
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
    Eigen::Matrix3d J;
    J << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1);
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(affine_geometry(J, origin));

    for (const auto& xi : parameters) {
        const auto point = reference.evaluate(xi);
        EXPECT_TRUE(point.position.isApprox(origin + J * xi, 1e-13));
        EXPECT_TRUE(point.jacobian.isApprox(J, 1e-13));
        EXPECT_NEAR(point.jacobian_determinant, J.determinant(), 1e-13);
    }
}

TEST(VolumeReferenceGeometry, ArbitraryPointGradientsMatchPhysicalFiniteDifferencesOnWarpedHex) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(warped_geometry());
    constexpr double h = 1e-6;
    for (const auto& xi : parameters) {
        const Eigen::Vector3d position = warped_position(xi);
        const auto point = reference.evaluate(xi);
        EXPECT_TRUE(point.position.isApprox(position, 1e-14));
        EXPECT_NEAR(point.jacobian_determinant, 1.0 - alpha * beta * xi.z() * xi.z(), 1e-14);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = position;
            Eigen::Vector3d minus = position;
            plus[d] += h;
            minus[d] -= h;
            const auto values_plus = litepgo::HexQ1Basis::evaluate(warped_parameter(plus)).values;
            const auto values_minus = litepgo::HexQ1Basis::evaluate(warped_parameter(minus)).values;
            const Eigen::Matrix<double, 8, 1> derivative = (values_plus - values_minus) / (2.0 * h);
            EXPECT_TRUE((point.basis_gradients.col(d) - derivative).isZero(2e-9));
        }
    }
}

TEST(VolumeReferenceGeometry, OwnsGeometryCoefficients) {
    Eigen::Matrix<double, 8, 3> X = corners();
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(X);
    X.setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto point = reference.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4));
    EXPECT_NEAR(point.jacobian_determinant, 1.0, 1e-14);
    EXPECT_TRUE(point.basis_gradients.allFinite());
}

TEST(VolumeReferenceGeometry, RejectsNonFiniteGeometry) {
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        Eigen::Matrix<double, 8, 3> X = corners();
        X(4, 2) = value;
        EXPECT_THROW((litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>(X)), std::invalid_argument);
    }
}

TEST(VolumeReferenceGeometry, RejectsNonFiniteParameters) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(corners());
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d xi(0.2, 0.3, 0.4);
            xi[d] = value;
            EXPECT_THROW(reference.evaluate(xi), std::invalid_argument);
        }
    }
}

TEST(VolumeReferenceGeometry, RejectsCollapsedAndInvertedGeometryWhenEvaluating) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> collapsed(Eigen::Matrix<double, 8, 3>::Zero());
    EXPECT_THROW(collapsed.evaluate(xi), std::invalid_argument);
    Eigen::Matrix<double, 8, 3> X = corners();
    X.col(2).setZero();
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> flat(X);
    EXPECT_THROW(flat.evaluate(xi), std::invalid_argument);
    X = corners();
    X.col(0) *= -1.0;
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> inverted(X);
    EXPECT_THROW(inverted.evaluate(xi), std::invalid_argument);
}

TEST(VolumeReferenceGeometry, ChecksJacobianAtTheRequestedPoint) {
    Eigen::Matrix<double, 8, 3> X = corners();
    // det(J0) = 1 - 1.5*xi: validity depends on the requested parameter point.
    for (int i = 0; i < 8; ++i) {
        X(i, 2) *= 1.0 - 1.5 * X(i, 0);
    }
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> reference(X);
    EXPECT_NO_THROW(reference.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4)));
    EXPECT_THROW(reference.evaluate(Eigen::Vector3d(0.8, 0.3, 0.4)), std::invalid_argument);
}

TEST(VolumeReferenceGeometry, AcceptsSmallPositiveJacobiansAndRejectsOverflowWhenEvaluating) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> small(1e-6 * corners());
    const auto point = small.evaluate(xi);
    EXPECT_GT(point.jacobian_determinant, 0.0);
    EXPECT_TRUE(point.basis_gradients.allFinite());
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> enormous(1e200 * corners());
    EXPECT_THROW(enormous.evaluate(xi), std::invalid_argument);
}
