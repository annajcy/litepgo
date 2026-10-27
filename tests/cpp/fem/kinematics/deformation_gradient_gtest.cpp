#include "litepgo/fem/basis/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/basis/tet_p1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/deformation_gradient.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace {
using namespace litepgo;
Eigen::Matrix<double, 8, 3> nodes() {
    Eigen::Matrix<double, 8, 3> X_local;
    for (int a = 0; a < 8; ++a)
        X_local.row(a) = HexQ1Basis::reference_node(a).transpose();
    return X_local;
}
template <class B> void check_derivative() {
    auto X_local = nodes();
    X_local(6, 0) += 0.3;
    const InterpolatedField<HexQ1Basis> X_field(X_local);
    Eigen::MatrixXd u_local(B::num_basis_functions, 3);
    for (int j = 0; j < u_local.size(); ++j)
        u_local(j / 3, j % 3) = 0.02 * std::sin(j + 0.3);
    const auto deformation = DeformationGradient(InterpolatedField<B>(u_local), X_field);
    for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(-0.2, 1.1, 0.1)}) {
        const auto deformation_sample = deformation.evaluate(xi);
        const auto derivative = deformation_sample.dF_du_local();
        const Eigen::MatrixXd previous =
            Eigen::kroneckerProduct(deformation_sample.reference_mapping().dxi_dX().transpose(),
                                    Eigen::Matrix3d::Identity()) *
            InterpolatedField<B>(u_local).evaluate(xi).dgrad_xi_dcoeffs();
        EXPECT_TRUE(derivative.isApprox(previous, 1e-14));
        for (int j = 0; j < u_local.size(); ++j) {
            auto plus = u_local, minus = u_local;
            plus(j / 3, j % 3) += 1e-5;
            minus(j / 3, j % 3) -= 1e-5;
            const Eigen::Matrix3d fd = (DeformationGradient(InterpolatedField<B>(plus), X_field).evaluate(xi).F() -
                                        DeformationGradient(InterpolatedField<B>(minus), X_field).evaluate(xi).F()) /
                                       2e-5;
            EXPECT_LT((fd.reshaped() - derivative.col(j)).norm(), 1e-9 * (1 + derivative.col(j).norm()));
        }
    }
}
TEST(DeformationGradient, CoefficientDerivativesMatchPreviousFormulaAndFiniteDifferencesForEveryBasis) {
    check_derivative<HexQ1Basis>();
    check_derivative<TetP1Basis>();
    check_derivative<HexHermiteBasis>();
}
TEST(DeformationGradient, SamplesOwnSnapshotsAndCanOnlyBeConstructedByEvaluation) {
    const Eigen::Matrix3d A = (Eigen::Matrix3d() << 0.1, 0.2, 0, -0.3, 0.1, 0.2, 0, 0, 0.3).finished();
    auto X_local = nodes();
    Eigen::MatrixXd u_local = X_local * A.transpose();
    const auto deformation =
        DeformationGradient(InterpolatedField<HexQ1Basis>(u_local), InterpolatedField<HexQ1Basis>(X_local));
    using DS = decltype(deformation.evaluate(Eigen::Vector3d::Zero()));
    using US = InterpolatedField<HexQ1Basis>::Sample;
    static_assert(!std::is_default_constructible_v<DS>);
    static_assert(!std::is_constructible_v<DS, US, ReferenceMapping>);
    X_local.setZero();
    u_local.setZero();
    const auto detached = [&] {
        auto copy = deformation;
        return copy.evaluate({0.2, 0.3, 0.4});
    }();
    EXPECT_EQ(detached.xi(), Eigen::Vector3d(0.2, 0.3, 0.4));
    EXPECT_TRUE(detached.F().isApprox(Eigen::Matrix3d::Identity() + A, 1e-14));
    EXPECT_EQ(detached.dF_du_local().cols(), 24);
    const auto other = deformation.evaluate({0.8, 0.7, 0.6});
    EXPECT_EQ(detached.xi(), Eigen::Vector3d(0.2, 0.3, 0.4));
    EXPECT_NE(detached.xi(), other.xi());
}
TEST(DeformationGradient, ValidatesShapesParametersGeometryAndOverflow) {
    const auto X_field = InterpolatedField<HexQ1Basis>(nodes());
    EXPECT_THROW((DeformationGradient(InterpolatedField<HexQ1Basis>(Eigen::MatrixXd::Zero(8, 2)), X_field)),
                 std::invalid_argument);
    const auto deformation = DeformationGradient(InterpolatedField<HexQ1Basis>(Eigen::MatrixXd::Zero(8, 3)), X_field);
    EXPECT_THROW(deformation.evaluate(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
    EXPECT_THROW((DeformationGradient(InterpolatedField<HexQ1Basis>(nodes()),
                                      InterpolatedField<HexQ1Basis>(Eigen::MatrixXd::Zero(8, 3)))
                      .evaluate({0.2, 0.3, 0.4})),
                 std::invalid_argument);
    const auto tiny = InterpolatedField<HexQ1Basis>(nodes() * Eigen::Vector3d(1e-200, 1e100, 1e100).asDiagonal());
    const auto enormous = InterpolatedField<HexQ1Basis>(1e200 * nodes());
    const auto deformation_sample = DeformationGradient(enormous, tiny).evaluate({0.2, 0.3, 0.4});
    EXPECT_THROW(deformation_sample.F(), std::invalid_argument);
    const auto large_inverse =
        InterpolatedField<HexQ1Basis>(nodes() * Eigen::Vector3d(1e-308, 1e154, 1e154).asDiagonal());
    const auto overflow_derivative =
        DeformationGradient(InterpolatedField<HexQ1Basis>(Eigen::MatrixXd::Zero(8, 3)), large_inverse)
            .evaluate({2, 2, 2});
    EXPECT_THROW(overflow_derivative.dF_du_local(), std::invalid_argument);
}
TEST(DeformationGradient, SpatiallyConstantDisplacementHasZeroCoefficientGradient) {
    const auto deformation_sample = DeformationGradient(IdentityField<Eigen::Vector3d>(Eigen::Vector3d(1, 2, 3)),
                                                        InterpolatedField<HexQ1Basis>(nodes()))
                                        .evaluate({0.2, 0.3, 0.4});
    EXPECT_EQ(deformation_sample.F(), Eigen::Matrix3d::Identity());
    EXPECT_TRUE(deformation_sample.dF_du_local().isZero());
}
TEST(DeformationGradient, DifferentDisplacementBasisOnWarpedCoordinatesUsesPhysicalChainRule) {
    using TensorBasis = TensorProductBasis3D<LagrangeBasis>;
    const auto warped = [](const Eigen::Vector3d& xi) -> Eigen::Vector3d {
        return {xi.x() + 0.4 * xi.y() * xi.z(), xi.y() + 0.2 * xi.x() * xi.z(), xi.z()};
    };
    const auto inverse = [](const Eigen::Vector3d& X) -> Eigen::Vector3d {
        const double det = 1 - 0.08 * X.z() * X.z();
        return {(X.x() - 0.4 * X.z() * X.y()) / det, (X.y() - 0.2 * X.z() * X.x()) / det, X.z()};
    };
    auto X_local = nodes();
    Eigen::Matrix<double, 8, 3> u_local;
    for (int a = 0; a < 8; ++a) {
        X_local.row(a) = warped(HexQ1Basis::reference_node(a));
        const Eigen::Vector3d xi(a / 4, (a / 2) % 2, a % 2);
        u_local.row(a) << xi.x() * xi.y(), xi.y() * xi.z(), xi.z() * xi.x();
    }
    const auto u_field = InterpolatedField<TensorBasis>(u_local);
    const auto deformation = DeformationGradient(u_field, InterpolatedField<HexQ1Basis>(X_local));
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const auto point = deformation.evaluate(xi);
    const auto X = warped(xi);
    for (int d = 0; d < 3; ++d) {
        Eigen::Vector3d plus = X, minus = X;
        plus[d] += 1e-5;
        minus[d] -= 1e-5;
        const Eigen::Vector3d fd = (u_field.value(inverse(plus)) - u_field.value(inverse(minus))) / 2e-5;
        EXPECT_TRUE(fd.isApprox((point.F() - Eigen::Matrix3d::Identity()).col(d), 3e-9));
    }
}
} // namespace
