#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/geometry/rest.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

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

TEST(RestPoint, MapsUnitCubeAtArbitraryParameters) {
    const litepgo::Interpolation<litepgo::HexQ1Basis> reference(corners());
    for (const auto& xi : parameters) {
        const auto point = litepgo::rest_point(reference, xi);
        EXPECT_TRUE(point.position.isApprox(xi, 1e-14));
        EXPECT_TRUE(point.jacobian.isApprox(Eigen::Matrix3d::Identity(), 1e-14));
        EXPECT_EQ(point.xi, xi);
        EXPECT_TRUE(point.inverse_jacobian.isIdentity(1e-14));
        EXPECT_NEAR(point.jacobian_determinant, 1.0, 1e-14);
    }
}

TEST(RestPoint, MapsTranslatedBox) {
    const Eigen::Vector3d lengths(2.0, 3.0, 4.0);
    const Eigen::Matrix3d expected_jacobian = lengths.asDiagonal();
    const Eigen::Vector3d origin(1.0, -2.0, 3.0);
    const litepgo::Interpolation<litepgo::HexQ1Basis> reference(affine_geometry(expected_jacobian, origin));
    for (const auto& xi : parameters) {
        const auto point = litepgo::rest_point(reference, xi);
        EXPECT_TRUE(point.position.isApprox(origin + lengths.asDiagonal() * xi, 1e-14));
        EXPECT_TRUE(point.jacobian.isApprox(expected_jacobian, 1e-14));
        EXPECT_TRUE(point.inverse_jacobian.isApprox(expected_jacobian.inverse(), 1e-14));
        EXPECT_NEAR(point.jacobian_determinant, 24.0, 1e-13);
    }
}

TEST(RestPoint, MapsShearedHexAtArbitraryParameters) {
    Eigen::Matrix3d J;
    J << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1);
    const litepgo::Interpolation<litepgo::HexQ1Basis> reference(affine_geometry(J, origin));

    for (const auto& xi : parameters) {
        const auto point = litepgo::rest_point(reference, xi);
        EXPECT_TRUE(point.position.isApprox(origin + J * xi, 1e-13));
        EXPECT_TRUE(point.jacobian.isApprox(J, 1e-13));
        EXPECT_TRUE(point.inverse_jacobian.isApprox(J.inverse(), 1e-13));
        EXPECT_NEAR(point.jacobian_determinant, J.determinant(), 1e-13);
    }
}

TEST(RestPoint, InverseJacobianMatchesPhysicalFiniteDifferencesOnWarpedHex) {
    const litepgo::Interpolation<litepgo::HexQ1Basis> reference(warped_geometry());
    constexpr double h = 1e-6;
    for (const auto& xi : parameters) {
        const Eigen::Vector3d position = warped_position(xi);
        const auto point = litepgo::rest_point(reference, xi);
        EXPECT_TRUE(point.position.isApprox(position, 1e-14));
        EXPECT_NEAR(point.jacobian_determinant, 1.0 - alpha * beta * xi.z() * xi.z(), 1e-14);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = position;
            Eigen::Vector3d minus = position;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Vector3d derivative = (warped_parameter(plus) - warped_parameter(minus)) / (2.0 * h);
            EXPECT_TRUE((point.inverse_jacobian.col(d) - derivative).isZero(2e-9));
        }
    }
}

TEST(RestPoint, OwnsGeometryCoefficients) {
    Eigen::Matrix<double, 8, 3> X = corners();
    const litepgo::Interpolation<litepgo::HexQ1Basis> reference(X);
    X.setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto point = litepgo::rest_point(reference, Eigen::Vector3d(0.2, 0.3, 0.4));
    EXPECT_NEAR(point.jacobian_determinant, 1.0, 1e-14);
    EXPECT_TRUE(point.inverse_jacobian.allFinite());
}

TEST(RestPoint, RejectsNonFiniteGeometry) {
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        Eigen::Matrix<double, 8, 3> X = corners();
        X(4, 2) = value;
        EXPECT_THROW((litepgo::Interpolation<litepgo::HexQ1Basis>(X)), std::invalid_argument);
    }
}

TEST(RestPoint, RejectsNonFiniteParameters) {
    const litepgo::Interpolation<litepgo::HexQ1Basis> reference(corners());
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d xi(0.2, 0.3, 0.4);
            xi[d] = value;
            EXPECT_THROW(litepgo::rest_point(reference, xi), std::invalid_argument);
        }
    }
}

TEST(RestPoint, RejectsCollapsedAndInvertedGeometryWhenEvaluating) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::Interpolation<litepgo::HexQ1Basis> collapsed(Eigen::Matrix<double, 8, 3>::Zero());
    EXPECT_THROW(litepgo::rest_point(collapsed, xi), std::invalid_argument);
    Eigen::Matrix<double, 8, 3> X = corners();
    X.col(2).setZero();
    const litepgo::Interpolation<litepgo::HexQ1Basis> flat(X);
    EXPECT_THROW(litepgo::rest_point(flat, xi), std::invalid_argument);
    X = corners();
    X.col(0) *= -1.0;
    const litepgo::Interpolation<litepgo::HexQ1Basis> inverted(X);
    EXPECT_THROW(litepgo::rest_point(inverted, xi), std::invalid_argument);
}

TEST(RestPoint, ChecksJacobianAtTheRequestedPoint) {
    Eigen::Matrix<double, 8, 3> X = corners();
    // det(J0) = 1 - 1.5*xi: validity depends on the requested parameter point.
    for (int i = 0; i < 8; ++i) {
        X(i, 2) *= 1.0 - 1.5 * X(i, 0);
    }
    const litepgo::Interpolation<litepgo::HexQ1Basis> reference(X);
    EXPECT_NO_THROW(litepgo::rest_point(reference, Eigen::Vector3d(0.2, 0.3, 0.4)));
    EXPECT_THROW(litepgo::rest_point(reference, Eigen::Vector3d(0.8, 0.3, 0.4)), std::invalid_argument);
}

TEST(RestPoint, AcceptsSmallPositiveJacobiansAndRejectsOverflowWhenEvaluating) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::Interpolation<litepgo::HexQ1Basis> small(1e-6 * corners());
    const auto point = litepgo::rest_point(small, xi);
    EXPECT_GT(point.jacobian_determinant, 0.0);
    EXPECT_TRUE(point.inverse_jacobian.allFinite());
    const litepgo::Interpolation<litepgo::HexQ1Basis> enormous(1e200 * corners());
    EXPECT_THROW(litepgo::rest_point(enormous, xi), std::invalid_argument);
}

TEST(RestPoint, RejectsNonVolumeComponentsAndNonFiniteInverse) {
    for (int components : {1, 2, 4}) {
        const litepgo::Interpolation<litepgo::HexQ1Basis> geometry(Eigen::MatrixXd::Zero(8, components));
        EXPECT_THROW(litepgo::rest_point(geometry, Eigen::Vector3d::Zero()), std::invalid_argument);
    }
    const litepgo::Interpolation<litepgo::HexQ1Basis> geometry(corners() *
                                                               Eigen::Vector3d(1e-310, 1e100, 1e100).asDiagonal());
    EXPECT_THROW(litepgo::rest_point(geometry, Eigen::Vector3d::Constant(0.5)), std::invalid_argument);
}
