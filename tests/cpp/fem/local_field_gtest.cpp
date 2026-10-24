#include "litepgo/fem/constant_field.hpp"
#include "litepgo/fem/deformation_gradient_field.hpp"
#include "litepgo/fem/element_field.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/interpolated_value_field.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
template <class ValueType>
concept CanFormDirectField = requires { typename ElementField<ValueType>; };
static_assert(!CanFormDirectField<Eigen::Matrix<double, 0, 1>>);
static_assert(CanFormDirectField<Eigen::Vector2d>);

VolumeReferenceGeometry<HexQ1Basis>::GeometryCoefficients nodes() {
    VolumeReferenceGeometry<HexQ1Basis>::GeometryCoefficients X;
    for (int i = 0; i < 8; ++i)
        X.row(i) = HexQ1Basis::reference_node(i).transpose();
    return X;
}
TEST(LocalField, AffineValuesAndCoefficientJacobians) {
    const auto X = nodes();
    const VolumeReferenceGeometry<HexQ1Basis> geometry(X);
    const auto point = geometry.evaluate({0.2, 0.3, 0.4});
    Eigen::Matrix3d A;
    A << 0.1, 0.2, 0, -0.3, 0.1, 0.2, 0.1, 0, 0.3;
    Eigen::Matrix<double, 8, 3> u = X * A.transpose();
    Eigen::Matrix<double, 8, 2> c;
    c.col(0) = 2 + X.col(0).array();
    c.col(1) = 3 - X.col(1).array();
    const DeformationGradientField<HexQ1Basis> displacement(u);
    const InterpolatedValueField<HexQ1Basis, 2> parameters(c);
    u.setZero();
    EXPECT_TRUE(displacement.value(point).isApprox(Eigen::Matrix3d::Identity() + A, 1e-13));
    EXPECT_TRUE(parameters.value(point).isApprox(Eigen::Vector2d(2.2, 2.7), 1e-13));
    constexpr double h = 1e-4;
    u = X * A.transpose();
    for (int j = 0; j < 24; ++j) {
        auto plus = u, minus = u;
        plus(j / 3, j % 3) += h;
        minus(j / 3, j % 3) -= h;
        const Eigen::Matrix3d fd = (DeformationGradientField<HexQ1Basis>(plus).value(point) -
                                    DeformationGradientField<HexQ1Basis>(minus).value(point)) /
                                   (2 * h);
        EXPECT_TRUE(DeformationGradientField<HexQ1Basis>::jacobian(point).col(j).isApprox(
            Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fd.data()), 1e-9));
    }
    for (int j = 0; j < 16; ++j) {
        auto plus = c, minus = c;
        plus(j / 2, j % 2) += h;
        minus(j / 2, j % 2) -= h;
        EXPECT_TRUE((InterpolatedValueField<HexQ1Basis, 2>::jacobian(point).col(j).isApprox(
            (InterpolatedValueField<HexQ1Basis, 2>(plus).value(point) -
             InterpolatedValueField<HexQ1Basis, 2>(minus).value(point)) /
                (2 * h),
            1e-9)));
    }
    u(0, 0) = std::numeric_limits<double>::quiet_NaN();
    c(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((DeformationGradientField<HexQ1Basis>(u)), std::invalid_argument);
    EXPECT_THROW((InterpolatedValueField<HexQ1Basis, 2>(c)), std::invalid_argument);
}
TEST(LocalField, DirectValuesIgnoreLocationAndOwnData) {
    Eigen::Matrix3d input;
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    const ElementField<Eigen::Matrix3d> field(input);
    input.setZero();
    EXPECT_EQ(field.value(0.2), field.value(100.0));
    EXPECT_EQ(field.value(0.2)(2, 2), 9);
    EXPECT_TRUE(field.jacobian(0).isIdentity());
    const auto original = field.value(0);
    for (int j = 0; j < 9; ++j) {
        auto perturbed = original;
        perturbed.data()[j] += 1e-4;
        const Eigen::Matrix3d fd = (ElementField<Eigen::Matrix3d>(perturbed).value(0) - original) / 1e-4;
        EXPECT_LT((Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fd.data()) - field.jacobian(0).col(j)).norm(), 1e-10);
    }
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> row_major = original;
    EXPECT_EQ((ElementField<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(row_major).value(0)), original);
    input(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((ElementField<Eigen::Matrix3d>(input)), std::invalid_argument);
}
} // namespace

TEST(LocalField, DataConstructorsOwnOnlySelectedSnapshots) {
    litepgo::ElementDofData<Eigen::Matrix3d> data(2);
    data.set_value(0, Eigen::Matrix3d::Identity());
    data.set_value(1, 2 * Eigen::Matrix3d::Identity());
    const litepgo::ElementField first(data, 0);
    const litepgo::ElementField second(data, 1);
    static_assert(decltype(first)::num_dofs == 9);
    EXPECT_EQ(data.num_dofs(), 18);
    data.set_value(0, 3 * Eigen::Matrix3d::Identity());
    EXPECT_EQ(first.value(0), Eigen::Matrix3d::Identity());
    EXPECT_EQ(second.value(100), 2 * Eigen::Matrix3d::Identity());
    EXPECT_THROW((litepgo::ElementField(data, -1)), std::out_of_range);
    EXPECT_THROW((litepgo::ElementField(data, 2)), std::out_of_range);
    data.values()[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::ElementField(data, 0)), std::invalid_argument);
    const auto owned = [] {
        litepgo::ElementDofData<Eigen::Vector2d> temporary(1);
        temporary.set_value(0, {4, 5});
        return litepgo::ElementField(temporary, 0);
    }();
    EXPECT_EQ(owned.value(0), Eigen::Vector2d(4, 5));

    litepgo::ConstantDofData<Eigen::Vector2d> shared(Eigen::Vector2d(2, 3));
    const litepgo::ConstantField snapshot(shared);
    shared.set_value({4, 5});
    EXPECT_EQ(snapshot.value(0), Eigen::Vector2d(2, 3));
    EXPECT_EQ(snapshot.value(0), snapshot.value(100));
    EXPECT_TRUE(snapshot.jacobian(0).isIdentity());
    const auto shared_owned = [] {
        const litepgo::ConstantDofData<Eigen::Vector2d> temporary(Eigen::Vector2d(6, 7));
        return litepgo::ConstantField(temporary);
    }();
    EXPECT_EQ(shared_owned.value(0), Eigen::Vector2d(6, 7));
    shared.values()[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((litepgo::ConstantField(shared)), std::invalid_argument);
}

TEST(LocalField, ConstantMatrixJacobianUsesColumnMajorDofs) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> input;
    input << 1, 2, 3, 4;
    const litepgo::ConstantDofData data(input);
    const litepgo::ConstantField field(data);
    EXPECT_EQ(field.value(0), input);
    for (int j = 0; j < 4; ++j) {
        auto plus = data, minus = data;
        plus.values()[j] += 1e-4;
        minus.values()[j] -= 1e-4;
        const Eigen::Matrix2d fd =
            (litepgo::ConstantField(plus).value(0) - litepgo::ConstantField(minus).value(0)) / 2e-4;
        EXPECT_LT((Eigen::Map<const Eigen::Vector4d>(fd.data()) - field.jacobian(0).col(j)).norm(), 1e-10);
    }
}
