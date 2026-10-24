#include "litepgo/fem/direct_local_field.hpp"
#include "litepgo/fem/displacement_gradient_field.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/interpolated_value_field.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
using B = HexQ1Basis;
using G = VolumeReferenceGeometry<B>;
using D = DisplacementGradientField<B>;
using V = InterpolatedValueField<B, 2>;

G::GeometryCoefficients nodes() {
    G::GeometryCoefficients X;
    for (int i = 0; i < 8; ++i)
        X.row(i) = B::reference_node(i).transpose();
    return X;
}
TEST(LocalField, AffineValuesAndCoefficientJacobians) {
    const auto X = nodes();
    const G geometry(X);
    const auto point = geometry.evaluate({0.2, 0.3, 0.4});
    Eigen::Matrix3d A;
    A << 0.1, 0.2, 0, -0.3, 0.1, 0.2, 0.1, 0, 0.3;
    D::Coefficients u = X * A.transpose();
    V::Coefficients c;
    c.col(0) = 2 + X.col(0).array();
    c.col(1) = 3 - X.col(1).array();
    const D displacement(u);
    const V parameters(c);
    u.setZero();
    EXPECT_TRUE(displacement.value(point).isApprox(Eigen::Matrix3d::Identity() + A, 1e-13));
    EXPECT_TRUE(parameters.value(point).isApprox(Eigen::Vector2d(2.2, 2.7), 1e-13));
    constexpr double h = 1e-4;
    u = X * A.transpose();
    for (int j = 0; j < 24; ++j) {
        auto plus = u, minus = u;
        plus(j / 3, j % 3) += h;
        minus(j / 3, j % 3) -= h;
        const Eigen::Matrix3d fd = (D(plus).value(point) - D(minus).value(point)) / (2 * h);
        EXPECT_TRUE(D::jacobian(point).col(j).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fd.data()), 1e-9));
    }
    for (int j = 0; j < 16; ++j) {
        auto plus = c, minus = c;
        plus(j / 2, j % 2) += h;
        minus(j / 2, j % 2) -= h;
        EXPECT_TRUE(V::jacobian(point).col(j).isApprox((V(plus).value(point) - V(minus).value(point)) / (2 * h), 1e-9));
    }
    u(0, 0) = std::numeric_limits<double>::quiet_NaN();
    c(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((D(u)), std::invalid_argument);
    EXPECT_THROW((V(c)), std::invalid_argument);
}
TEST(LocalField, DirectValuesIgnoreLocationAndOwnData) {
    Eigen::Matrix3d input;
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    const DirectLocalField<Eigen::Matrix3d> field(input);
    input.setZero();
    EXPECT_EQ(field.value(0.2), field.value(100.0));
    EXPECT_EQ(field.value(0.2)(2, 2), 9);
    EXPECT_TRUE(field.jacobian(0).isIdentity());
    const auto original = field.value(0);
    for (int j = 0; j < 9; ++j) {
        auto perturbed = original;
        perturbed.data()[j] += 1e-4;
        const Eigen::Matrix3d fd = (DirectLocalField<Eigen::Matrix3d>(perturbed).value(0) - original) / 1e-4;
        EXPECT_LT((Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fd.data()) - field.jacobian(0).col(j)).norm(), 1e-10);
    }
    using RowMatrix = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    const RowMatrix row_major = original;
    EXPECT_EQ(DirectLocalField<RowMatrix>(row_major).value(0), original);
    const DirectLocalField<Eigen::Matrix<double, 0, 1>> empty({});
    EXPECT_EQ(empty.value(0).size(), 0);
    EXPECT_EQ(empty.jacobian(0).size(), 0);
    input(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((DirectLocalField<Eigen::Matrix3d>(input)), std::invalid_argument);
}
} // namespace
