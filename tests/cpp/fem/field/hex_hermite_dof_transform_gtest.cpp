#include "litepgo/fem/field/hex_hermite_dof_transform.hpp"

#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
using namespace litepgo;

Eigen::Matrix<double, 8, 3> box_nodes(const Eigen::Vector3d& lengths,
                                      const Eigen::Vector3d& origin = Eigen::Vector3d::Zero()) {
    Eigen::Matrix<double, 8, 3> X_nodes;
    for (int v = 0; v < 8; ++v)
        X_nodes.row(v) = (origin + lengths.cwiseProduct(HexQ1Basis::reference_node(v))).transpose();
    return X_nodes;
}

TEST(HexHermiteDofTransform, ScalesEveryPhysicalDerivativeSlotAndComponent) {
    const auto X_nodes = box_nodes({2, 3, 5}, {0.2, -0.3, 0.4});
    const auto S = hex_hermite_dof_transform(X_nodes);
    ASSERT_EQ(S.rows(), 192);
    ASSERT_EQ(S.cols(), 192);
    const std::array<double, 8> expected{1, 2, 3, 5, 6, 10, 15, 30};
    for (int v = 0; v < 8; ++v)
        for (int slot = 0; slot < 8; ++slot)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(S.diagonal()[24 * v + 3 * slot + c], expected[static_cast<std::size_t>(slot)], 1e-13);
    EXPECT_TRUE(hex_hermite_dof_transform(box_nodes(Eigen::Vector3d::Ones())).diagonal().isOnes());
}

TEST(HexHermiteDofTransform, RejectsInvalidNodeDimensionsAndNonFiniteCoordinates) {
    for (const auto shape :
         {std::array{0, 3}, std::array{7, 3}, std::array{9, 3}, std::array{8, 0}, std::array{8, 2}, std::array{8, 4}}) {
        SCOPED_TRACE(testing::PrintToString(shape));
        EXPECT_THROW(hex_hermite_dof_transform(Eigen::MatrixXd::Zero(shape[0], shape[1])), std::invalid_argument);
    }
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()})
        for (int v = 0; v < 8; ++v)
            for (int d = 0; d < 3; ++d) {
                auto X_nodes = box_nodes({2, 3, 5});
                X_nodes(v, d) = bad;
                EXPECT_THROW(hex_hermite_dof_transform(X_nodes), std::invalid_argument);
            }
}

TEST(HexHermiteDofTransform, ReproducesPhysicalPolynomialWithMixedDerivatives) {
    const auto X_nodes = box_nodes({2, 3, 5}, {0.2, -0.3, 0.4});
    const InterpolatedField3D<HexQ1Basis> X_field(X_nodes);
    const auto S = hex_hermite_dof_transform(X_nodes);
    const Eigen::Vector3d factors(0.7, -1.3, 2.1);
    Eigen::MatrixXd x_local(8, 24);
    for (int v = 0; v < 8; ++v) {
        const Eigen::Vector3d X = X_nodes.row(v).transpose();
        // Physical derivatives of X*Y*Z in the documented eight-slot order.
        const std::array<double, 8> derivatives{
            X.x() * X.y() * X.z(), X.y() * X.z(), X.x() * X.z(), X.x() * X.y(), X.z(), X.y(), X.x(), 1};
        for (int slot = 0; slot < 8; ++slot)
            x_local.row(v).segment<3>(3 * slot) = (factors * derivatives[static_cast<std::size_t>(slot)]).transpose();
    }
    const Eigen::VectorXd c = S * x_local.reshaped<Eigen::RowMajor>();
    const InterpolatedField3D<HexHermiteBasis> displacement(c.reshaped<Eigen::RowMajor>(64, 3));
    for (const Eigen::Vector3d xi :
         {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0.2, 0.3, 0.4)}) {
        const auto& X = X_field.value(xi);
        EXPECT_TRUE(displacement.value(xi).isApprox(factors * X.x() * X.y() * X.z(), 1e-12));
        const Eigen::Vector3d gradient(X.y() * X.z(), X.x() * X.z(), X.x() * X.y());
        EXPECT_TRUE((displacement.dvalue_dxi(xi) * X_field.dvalue_dxi(xi).inverse())
                        .isApprox(factors * gradient.transpose(), 1e-12));
    }
}

TEST(HexHermiteDofTransform, RejectsUnsupportedGeometryAndUnrepresentableScales) {
    for (int which = 0; which < 6; ++which) {
        SCOPED_TRACE(which);
        auto X_nodes = box_nodes({2, 3, 5});
        if (which == 0)
            X_nodes.col(0) += 0.1 * X_nodes.col(1);
        if (which == 1)
            X_nodes(6, 0) += 0.1;
        if (which == 2)
            X_nodes.col(2).setZero();
        if (which == 3 || which == 4)
            X_nodes.col(0) *= -1;
        if (which == 4)
            X_nodes.col(1) *= -1;
        if (which == 5) {
            const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            X_nodes = (X_nodes * rotation.transpose()).eval();
        }
        EXPECT_THROW(hex_hermite_dof_transform(X_nodes), std::invalid_argument);
    }
    for (const Eigen::Vector3d lengths :
         {Eigen::Vector3d(1e-200, 1e-200, 1e200), Eigen::Vector3d(1e200, 1e200, 1e-200),
          Eigen::Vector3d::Constant(1e-110).eval(), Eigen::Vector3d::Constant(1e120).eval()}) {
        EXPECT_THROW(hex_hermite_dof_transform(box_nodes(lengths)), std::invalid_argument);
    }
}

TEST(HexHermiteDofTransform, RejectsInconsistentCornersAndOverflowingEdgeLengths) {
    for (int v = 0; v < 8; ++v)
        for (int d = 0; d < 3; ++d) {
            auto X_nodes = box_nodes(Eigen::Vector3d::Ones());
            X_nodes(v, d) += 0.1;
            EXPECT_THROW(hex_hermite_dof_transform(X_nodes), std::invalid_argument);
        }
    auto swapped = box_nodes(Eigen::Vector3d::Ones());
    swapped.row(2).swap(swapped.row(3));
    EXPECT_THROW(hex_hermite_dof_transform(swapped), std::invalid_argument);
    auto duplicate = box_nodes(Eigen::Vector3d::Ones());
    duplicate.row(6) = duplicate.row(7);
    EXPECT_THROW(hex_hermite_dof_transform(duplicate), std::invalid_argument);
    auto overflowing = box_nodes(Eigen::Vector3d::Ones());
    for (int v = 0; v < 8; ++v)
        overflowing(v, 0) = HexHermiteBasis::reference_node(8 * v).x() == 0 ? -std::numeric_limits<double>::max()
                                                                            : std::numeric_limits<double>::max();
    ASSERT_TRUE(overflowing.allFinite());
    EXPECT_THROW(hex_hermite_dof_transform(overflowing), std::invalid_argument);
}
} // namespace
