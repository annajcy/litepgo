#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/reference_mapping.hpp"

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
    Eigen::Matrix<double, 8, 3> X_nodes;
    X_nodes << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return X_nodes;
}

Eigen::Matrix<double, 8, 3> affine_geometry(const Eigen::Matrix3d& dX_dxi, const Eigen::Vector3d& origin) {
    return (corners() * dX_dxi.transpose()).rowwise() + origin.transpose();
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
    Eigen::Matrix<double, 8, 3> X_nodes;
    for (int i = 0; i < 8; ++i) {
        X_nodes.row(i) = warped_position(nodes.row(i).transpose()).transpose();
    }
    return X_nodes;
}

} // namespace

TEST(ReferenceMapping, MapsUnitCubeAtArbitraryParameters) {
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> reference(corners());
    for (const auto& xi : parameters) {
        const auto point = litepgo::ReferenceMapping(reference.evaluate(xi));
        EXPECT_TRUE(point.X().isApprox(xi, 1e-14));
        EXPECT_TRUE(point.dX_dxi().isApprox(Eigen::Matrix3d::Identity(), 1e-14));
        EXPECT_EQ(point.xi(), xi);
        EXPECT_TRUE(point.dxi_dX().isIdentity(1e-14));
        EXPECT_NEAR(point.det_dX_dxi(), 1.0, 1e-14);
    }
}

TEST(ReferenceMapping, MapsTranslatedBox) {
    const Eigen::Vector3d lengths(2.0, 3.0, 4.0);
    const Eigen::Matrix3d expected_dX_dxi = lengths.asDiagonal();
    const Eigen::Vector3d origin(1.0, -2.0, 3.0);
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> reference(affine_geometry(expected_dX_dxi, origin));
    for (const auto& xi : parameters) {
        const auto point = litepgo::ReferenceMapping(reference.evaluate(xi));
        EXPECT_TRUE(point.X().isApprox(origin + lengths.asDiagonal() * xi, 1e-14));
        EXPECT_TRUE(point.dX_dxi().isApprox(expected_dX_dxi, 1e-14));
        EXPECT_TRUE(point.dxi_dX().isApprox(expected_dX_dxi.inverse(), 1e-14));
        EXPECT_NEAR(point.det_dX_dxi(), 24.0, 1e-13);
    }
}

TEST(ReferenceMapping, MapsShearedHexAtArbitraryParameters) {
    Eigen::Matrix3d dX_dxi;
    dX_dxi << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1);
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> reference(affine_geometry(dX_dxi, origin));

    for (const auto& xi : parameters) {
        const auto point = litepgo::ReferenceMapping(reference.evaluate(xi));
        EXPECT_TRUE(point.X().isApprox(origin + dX_dxi * xi, 1e-13));
        EXPECT_TRUE(point.dX_dxi().isApprox(dX_dxi, 1e-13));
        EXPECT_TRUE(point.dxi_dX().isApprox(dX_dxi.inverse(), 1e-13));
        EXPECT_NEAR(point.det_dX_dxi(), dX_dxi.determinant(), 1e-13);
    }
}

TEST(ReferenceMapping, InverseJacobianMatchesPhysicalFiniteDifferencesOnWarpedHex) {
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> reference(warped_geometry());
    constexpr double h = 1e-6;
    for (const auto& xi : parameters) {
        const Eigen::Vector3d position = warped_position(xi);
        const auto point = litepgo::ReferenceMapping(reference.evaluate(xi));
        EXPECT_TRUE(point.X().isApprox(position, 1e-14));
        EXPECT_NEAR(point.det_dX_dxi(), 1.0 - alpha * beta * xi.z() * xi.z(), 1e-14);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = position;
            Eigen::Vector3d minus = position;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Vector3d derivative = (warped_parameter(plus) - warped_parameter(minus)) / (2.0 * h);
            EXPECT_TRUE((point.dxi_dX().col(d) - derivative).isZero(2e-9));
        }
    }
}

TEST(ReferenceMapping, OwnsGeometryCoefficients) {
    Eigen::Matrix<double, 8, 3> X_nodes = corners();
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> reference(X_nodes);
    X_nodes.setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto point = litepgo::ReferenceMapping(reference.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4)));
    EXPECT_NEAR(point.det_dX_dxi(), 1.0, 1e-14);
    EXPECT_TRUE(point.dxi_dX().allFinite());
}

TEST(ReferenceMapping, RejectsNonFiniteGeometry) {
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        Eigen::Matrix<double, 8, 3> X_nodes = corners();
        X_nodes(4, 2) = value;
        EXPECT_THROW((litepgo::InterpolatedField<litepgo::HexQ1Basis>(X_nodes)), std::invalid_argument);
    }
}

TEST(ReferenceMapping, RejectsNonFiniteParameters) {
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> reference(corners());
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d xi(0.2, 0.3, 0.4);
            xi[d] = value;
            EXPECT_THROW(litepgo::ReferenceMapping(reference.evaluate(xi)), std::invalid_argument);
        }
    }
}

TEST(ReferenceMapping, RejectsCollapsedAndInvertedGeometryWhenEvaluating) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> collapsed(Eigen::Matrix<double, 8, 3>::Zero());
    EXPECT_THROW(litepgo::ReferenceMapping(collapsed.evaluate(xi)), std::invalid_argument);
    Eigen::Matrix<double, 8, 3> X_nodes = corners();
    X_nodes.col(2).setZero();
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> flat(X_nodes);
    EXPECT_THROW(litepgo::ReferenceMapping(flat.evaluate(xi)), std::invalid_argument);
    X_nodes = corners();
    X_nodes.col(0) *= -1.0;
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> inverted(X_nodes);
    EXPECT_THROW(litepgo::ReferenceMapping(inverted.evaluate(xi)), std::invalid_argument);
}

TEST(ReferenceMapping, ChecksJacobianAtTheRequestedPoint) {
    Eigen::Matrix<double, 8, 3> X_nodes = corners();
    // det(J0) = 1 - 1.5*xi: validity depends on the requested parameter point.
    for (int i = 0; i < 8; ++i) {
        X_nodes(i, 2) *= 1.0 - 1.5 * X_nodes(i, 0);
    }
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> reference(X_nodes);
    EXPECT_NO_THROW(litepgo::ReferenceMapping(reference.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4))));
    EXPECT_THROW(litepgo::ReferenceMapping(reference.evaluate(Eigen::Vector3d(0.8, 0.3, 0.4))), std::invalid_argument);
}

TEST(ReferenceMapping, AcceptsSmallPositiveJacobiansAndRejectsOverflowWhenEvaluating) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> small(1e-6 * corners());
    const auto point = litepgo::ReferenceMapping(small.evaluate(xi));
    EXPECT_GT(point.det_dX_dxi(), 0.0);
    EXPECT_TRUE(point.dxi_dX().allFinite());
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> enormous(1e200 * corners());
    EXPECT_THROW(litepgo::ReferenceMapping(enormous.evaluate(xi)), std::invalid_argument);
}

TEST(ReferenceMapping, RejectsNonVolumeComponentsAndNonFiniteInverse) {
    for (int components : {1, 2, 4}) {
        const litepgo::InterpolatedField<litepgo::HexQ1Basis> X_field(Eigen::MatrixXd::Zero(8, components));
        EXPECT_THROW(litepgo::ReferenceMapping(X_field.evaluate(Eigen::Vector3d::Zero())), std::invalid_argument);
    }
    const litepgo::InterpolatedField<litepgo::HexQ1Basis> X_field(corners() *
                                                                  Eigen::Vector3d(1e-310, 1e100, 1e100).asDiagonal());
    EXPECT_THROW(litepgo::ReferenceMapping(X_field.evaluate(Eigen::Vector3d::Constant(0.5))), std::invalid_argument);
}
