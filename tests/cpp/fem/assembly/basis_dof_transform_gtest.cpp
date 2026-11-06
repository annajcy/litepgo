#include "litepgo/fem/assembly/basis_dof_transform.hpp"
#include <memory>

#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace litepgo::fem::test {

namespace {

Eigen::Matrix<double, 8, 3> box_nodes(const Eigen::Vector3d& lengths,
                                      const Eigen::Vector3d& origin = Eigen::Vector3d::Zero()) {
    Eigen::Matrix<double, 8, 3> X_nodes;
    for (int v = 0; v < 8; ++v)
        X_nodes.row(v) = (origin + lengths.cwiseProduct(HexQ1Basis::coefficient_info(v).point)).transpose();
    return X_nodes;
}

auto make_transform(const Eigen::MatrixXd& nodes) {
    InterpolatedResponse<HexQ1Basis> geometry(nodes);
    return BasisDofTransform<HexHermiteBasis>(geometry);
}

TEST(HexHermiteDofTransform, ScalesEveryPhysicalDerivativeSlotAndComponent) {
    const auto X_nodes = box_nodes({2, 3, 5}, {0.2, -0.3, 0.4});
    const auto S = make_transform(X_nodes);
    const auto coefficients = S.to_coefficients(Eigen::MatrixXd::Ones(8, 24));
    const Eigen::VectorXd scales = coefficients.reshaped<Eigen::RowMajor>();
    const std::array<double, 8> expected{1, 2, 3, 5, 6, 10, 15, 30};
    for (int v = 0; v < 8; ++v)
        for (int slot = 0; slot < 8; ++slot)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(scales[24 * v + 3 * slot + c], expected[static_cast<std::size_t>(slot)], 1e-13);
    EXPECT_TRUE(
        make_transform(box_nodes(Eigen::Vector3d::Ones())).to_coefficients(Eigen::MatrixXd::Ones(8, 24)).isOnes());
}

TEST(HexHermiteDofTransform, RejectsInvalidNodeDimensionsAndNonFiniteCoordinates) {
    for (const auto shape :
         {std::array{0, 3}, std::array{7, 3}, std::array{9, 3}, std::array{8, 0}, std::array{8, 2}, std::array{8, 4}}) {
        SCOPED_TRACE(testing::PrintToString(shape));
        EXPECT_THROW(make_transform(Eigen::MatrixXd::Zero(shape[0], shape[1])), std::invalid_argument);
    }
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()})
        for (int v = 0; v < 8; ++v)
            for (int d = 0; d < 3; ++d) {
                auto X_nodes = box_nodes({2, 3, 5});
                X_nodes(v, d) = bad;
                EXPECT_THROW(make_transform(X_nodes), std::invalid_argument);
            }
}

TEST(HexHermiteDofTransform, ReproducesPhysicalPolynomialWithMixedDerivatives) {
    const auto X_nodes = box_nodes({2, 3, 5}, {0.2, -0.3, 0.4});
    auto X_field = std::make_shared<InterpolatedResponse<HexQ1Basis>>(X_nodes);
    const auto S = make_transform(X_nodes);
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
    auto displacement = std::make_shared<InterpolatedResponse<HexHermiteBasis>>(S.to_coefficients(x_local));
    for (const Eigen::Vector3d xi :
         {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0.2, 0.3, 0.4)}) {
        const auto& X = X_field->eval_value(xi);
        EXPECT_TRUE(displacement->eval_value(xi).isApprox(factors * X.x() * X.y() * X.z(), 1e-12));
        const Eigen::Vector3d gradient(X.y() * X.z(), X.x() * X.z(), X.x() * X.y());
        EXPECT_TRUE(
            (Eigen::MatrixXd(displacement->eval_jacobian(xi)) * Eigen::MatrixXd(X_field->eval_jacobian(xi)).inverse())
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
        EXPECT_THROW(make_transform(X_nodes), std::invalid_argument);
    }
    for (const Eigen::Vector3d lengths :
         {Eigen::Vector3d(1e-200, 1e-200, 1e200), Eigen::Vector3d(1e200, 1e200, 1e-200),
          Eigen::Vector3d::Constant(1e-110).eval(), Eigen::Vector3d::Constant(1e120).eval()}) {
        EXPECT_THROW(make_transform(box_nodes(lengths)), std::invalid_argument);
    }
}

TEST(HexHermiteDofTransform, RejectsInconsistentCornersAndOverflowingEdgeLengths) {
    for (int v = 0; v < 8; ++v)
        for (int d = 0; d < 3; ++d) {
            auto X_nodes = box_nodes(Eigen::Vector3d::Ones());
            X_nodes(v, d) += 0.1;
            EXPECT_THROW(make_transform(X_nodes), std::invalid_argument);
        }
    auto swapped = box_nodes(Eigen::Vector3d::Ones());
    swapped.row(2).swap(swapped.row(3));
    EXPECT_THROW(make_transform(swapped), std::invalid_argument);
    auto duplicate = box_nodes(Eigen::Vector3d::Ones());
    duplicate.row(6) = duplicate.row(7);
    EXPECT_THROW(make_transform(duplicate), std::invalid_argument);
    auto overflowing = box_nodes(Eigen::Vector3d::Ones());
    for (int v = 0; v < 8; ++v)
        overflowing(v, 0) = HexHermiteBasis::coefficient_info(8 * v).point.x() == 0
                                ? -std::numeric_limits<double>::max()
                                : std::numeric_limits<double>::max();
    ASSERT_TRUE(overflowing.allFinite());
    EXPECT_THROW(make_transform(overflowing), std::invalid_argument);
}

template <class Basis> void check_identity_ownership(int nodes) {
    auto X = std::make_shared<InterpolatedResponse<HexQ1Basis>>(box_nodes({2, 3, 5}));
    const BasisDofTransform<Basis> transform(*X);
    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(nodes, 3, 2);
    const auto& coefficients = transform.to_coefficients(input);
    input.setZero();
    EXPECT_TRUE(coefficients.isConstant(2));
    const auto& from_temporary = transform.to_coefficients(Eigen::MatrixXd::Constant(nodes, 3, 3));
    EXPECT_TRUE(from_temporary.isConstant(3));
}

TEST(BasisDofTransform, IdentityReturnsOwnedCoefficients) {
    check_identity_ownership<TetP1Basis>(4);
    check_identity_ownership<HexQ1Basis>(8);
}

template <class Transform> void check_pullback_dimensions(const Transform& transform, int n) {
    Eigen::VectorXd g = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd H = Eigen::MatrixXd::Ones(n, n);
    for (int size : {0, n - 1, n + 1}) {
        Eigen::VectorXd bad = Eigen::VectorXd::Ones(size);
        EXPECT_THROW(transform.pullback(&bad, &H), std::invalid_argument);
        EXPECT_TRUE(bad.isOnes());
        EXPECT_TRUE(H.isOnes());
        EXPECT_THROW(transform.pullback(&bad, nullptr), std::invalid_argument);
    }
    for (const auto shape : {std::array{0, 0}, std::array{n - 1, n}, std::array{n, n + 1}}) {
        Eigen::MatrixXd bad = Eigen::MatrixXd::Ones(shape[0], shape[1]);
        EXPECT_THROW(transform.pullback(&g, &bad), std::invalid_argument);
        EXPECT_TRUE(g.isOnes());
        EXPECT_TRUE(bad.isOnes());
        EXPECT_THROW(transform.pullback(nullptr, &bad), std::invalid_argument);
    }
    EXPECT_NO_THROW(transform.pullback(nullptr, nullptr));
    Eigen::VectorXd independent_g = g;
    Eigen::MatrixXd independent_H = H;
    transform.pullback(&independent_g, nullptr);
    transform.pullback(nullptr, &independent_H);
    transform.pullback(&g, &H);
    EXPECT_EQ(g, independent_g);
    EXPECT_EQ(H, independent_H);
}

TEST(BasisDofTransform, RejectsDerivativeShapesBeforeModifyingEitherOutput) {
    auto X = std::make_shared<InterpolatedResponse<HexQ1Basis>>(box_nodes({2, 3, 5}));
    check_pullback_dimensions(BasisDofTransform<TetP1Basis>(*X), 12);
    check_pullback_dimensions(BasisDofTransform<HexQ1Basis>(*X), 24);
    check_pullback_dimensions(BasisDofTransform<HexHermiteBasis>(*X), 192);
}
TEST(HexHermiteDofTransform, RowMajorPullbackMatchesDenseAndPreservesPadding) {
    const auto transform = make_transform(box_nodes({2, 3, 5}));
    const Eigen::MatrixXd original = Eigen::MatrixXd::Random(192, 192);
    Eigen::MatrixXd expected = original;
    transform.pullback(nullptr, &expected);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> storage =
        Eigen::MatrixXd::Constant(194, 195, -7);
    storage.block(1, 1, 192, 192) = original;
    Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<-1, -1>> block(storage.data() + storage.outerStride() + 1, 192, 192,
                                                                Eigen::Stride<-1, -1>(1, storage.outerStride()));
    StridedMatRef view(block);
    transform.pullback_into(nullptr, &view);
    EXPECT_EQ(block, expected);
    EXPECT_TRUE(storage.row(0).isConstant(-7));
    EXPECT_TRUE(storage.row(193).isConstant(-7));
    EXPECT_TRUE(storage.col(0).isConstant(-7));
    EXPECT_TRUE(storage.rightCols(2).isConstant(-7));
    block.setConstant(std::numeric_limits<double>::max());
    EXPECT_THROW(transform.pullback_into(nullptr, &view), std::domain_error);
}

} // namespace

} // namespace litepgo::fem::test
