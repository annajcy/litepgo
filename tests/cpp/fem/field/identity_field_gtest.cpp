#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

using namespace litepgo;

TEST(LocalField, DirectValuesIgnoreLocationAndOwnData) {
    Eigen::Matrix3d input;
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    const IdentityField3D field(input.reshaped());
    input.setZero();
    Eigen::Vector3d first_point, second_point;
    first_point.setConstant(0.2);
    second_point.setConstant(100.0);
    EXPECT_EQ(field.eval_value(first_point), field.eval_value(second_point));
    EXPECT_EQ(field.eval_value(Eigen::Vector3d::Zero())[8], 9);
    EXPECT_TRUE(field.eval_dvalue_dcoeffs(Eigen::Vector3d::Zero()).isIdentity());
    const auto original = field.eval_value(Eigen::Vector3d::Zero());
    for (int j = 0; j < 9; ++j) {
        auto perturbed = original;
        perturbed.data()[j] += 1e-4;
        const Eigen::VectorXd fd = (IdentityField3D(perturbed).eval_value(Eigen::Vector3d::Zero()) - original) / 1e-4;
        EXPECT_LT((fd - field.eval_dvalue_dcoeffs(Eigen::Vector3d::Zero()).col(j)).norm(), 1e-10);
    }
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> row_major = original.reshaped(3, 3);
    EXPECT_EQ(IdentityField3D(row_major.reshaped()).eval_value(Eigen::Vector3d::Zero()), original);
    input(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((IdentityField3D(input.reshaped())), std::invalid_argument);
}

TEST(LocalField, ExplicitlyExtractedValuesOwnOnlySelectedSnapshots) {
    litepgo::Vec data = Vec::Zero(18);
    const ElementDofMap map(TetTopology(4, {{0, 1, 2, 3}, {0, 1, 2, 3}}), 9);
    data.head<9>() = Eigen::Matrix3d::Identity().eval().reshaped();
    data.tail<9>() = (2 * Eigen::Matrix3d::Identity()).eval().reshaped();
    const litepgo::IdentityField3D first(map.gather(0, data));
    const litepgo::IdentityField3D second(map.gather(1, data));
    EXPECT_EQ(first.num_dofs(), 9);
    EXPECT_EQ(data.size(), 18);
    data.head<9>() = (3 * Eigen::Matrix3d::Identity()).eval().reshaped();
    EXPECT_EQ(first.eval_value(Eigen::Vector3d::Zero()), Eigen::Matrix3d::Identity().reshaped());
    EXPECT_EQ(second.eval_value(Eigen::Vector3d::Zero()), (2 * Eigen::Matrix3d::Identity()).reshaped());
    EXPECT_THROW((litepgo::IdentityField3D(map.gather(-1, data))), std::out_of_range);
    EXPECT_THROW((litepgo::IdentityField3D(map.gather(2, data))), std::out_of_range);
    data[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField3D(map.gather(0, data))), std::invalid_argument);
    const auto owned = [] {
        litepgo::Vec temporary = Vec::Zero(2);
        temporary << 4, 5;
        return litepgo::IdentityField3D(ElementDofMap(TetTopology(4, {{0, 1, 2, 3}}), 2).gather(0, temporary));
    }();
    EXPECT_EQ(owned.eval_value(Eigen::Vector3d::Zero()), Eigen::Vector2d(4, 5));

    litepgo::Vec shared(Eigen::Vector2d(2, 3));
    const litepgo::IdentityField3D snapshot(shared);
    shared << 4, 5;
    EXPECT_EQ(snapshot.eval_value(Eigen::Vector3d::Zero()), Eigen::Vector2d(2, 3));
    EXPECT_EQ(snapshot.eval_value(Eigen::Vector3d::Zero()), snapshot.eval_value(Eigen::Vector3d::Zero()));
    EXPECT_TRUE(snapshot.eval_dvalue_dcoeffs(Eigen::Vector3d::Zero()).isIdentity());
    const auto shared_owned = [] {
        const litepgo::Vec temporary(Eigen::Vector2d(6, 7));
        return litepgo::IdentityField3D(temporary);
    }();
    EXPECT_EQ(shared_owned.eval_value(Eigen::Vector3d::Zero()), Eigen::Vector2d(6, 7));
    shared[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((litepgo::IdentityField3D(shared)), std::invalid_argument);
}

TEST(LocalField, FlattenedMatrixJacobianUsesColumnMajorDofs) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> input;
    input << 1, 2, 3, 4;
    const litepgo::Vec data(input.reshaped().eval());
    const litepgo::IdentityField3D field(data);
    EXPECT_EQ(field.eval_value(Eigen::Vector3d::Zero()), Eigen::Vector4d(1, 3, 2, 4));
    for (int j = 0; j < 4; ++j) {
        auto plus = data, minus = data;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::VectorXd fd = (litepgo::IdentityField3D(plus).eval_value(Eigen::Vector3d::Zero()) -
                                    litepgo::IdentityField3D(minus).eval_value(Eigen::Vector3d::Zero())) /
                                   2e-4;
        EXPECT_LT((fd - field.eval_dvalue_dcoeffs(Eigen::Vector3d::Zero()).col(j)).norm(), 1e-10);
    }
}

TEST(LocalField, IdentityFieldReturnsDynamicVectorsAndValidatesValues) {
    const Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::VectorXd input = Eigen::VectorXd::LinSpaced(4, 1, 4);
    const litepgo::IdentityField3D field(input);
    static_assert(std::is_same_v<decltype(field.eval_value(point)), Eigen::VectorXd>);
    const litepgo::IdentityField3D fixed(Eigen::Vector2d(2, 3));
    static_assert(std::is_same_v<decltype(fixed.eval_value(point)), Eigen::VectorXd>);
    EXPECT_EQ(field.num_dofs(), 4);
    EXPECT_TRUE(field.eval_dvalue_dcoeffs(point).isIdentity());
    input.setZero();
    EXPECT_EQ(field.eval_value(point), Eigen::VectorXd::LinSpaced(4, 1, 4));
    for (int j = 0; j < 4; ++j) {
        auto plus = field.eval_value(point), minus = plus;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::VectorXd fd =
            (litepgo::IdentityField3D(plus).eval_value(point) - litepgo::IdentityField3D(minus).eval_value(point)) /
            2e-4;
        EXPECT_TRUE(fd.isApprox(field.eval_dvalue_dcoeffs(point).col(j), 1e-10));
    }
    EXPECT_THROW((litepgo::IdentityField3D{Eigen::VectorXd{}}), std::invalid_argument);
    input[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField3D{input}), std::invalid_argument);
    const Eigen::Vector3d invalid_xi = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(field.eval_value(invalid_xi), std::invalid_argument);
    EXPECT_THROW(field.eval_dvalue_dxi(invalid_xi), std::invalid_argument);
    EXPECT_THROW(field.eval_dvalue_dcoeffs(invalid_xi), std::invalid_argument);
    EXPECT_THROW(field.eval_dgrad_xi_dcoeffs(invalid_xi), std::invalid_argument);
}

namespace {
template <int Dim> void check_constant_spatial_derivatives() {
    static_assert(std::is_same_v<IdentityField3D, IdentityFieldT<3>>);
    static_assert(std::is_same_v<IdentityField2D, IdentityFieldT<2>>);
    const IdentityFieldT<Dim> concrete(Eigen::Vector3d(1, 2, 3));
    const FieldT<Dim>& field = concrete;
    using Coordinate = typename FieldT<Dim>::Coordinate;
    const Coordinate xi = Coordinate::Constant(0.3);
    EXPECT_EQ(field.eval_value(xi), Eigen::Vector3d(1, 2, 3));
    EXPECT_TRUE(field.eval_dvalue_dcoeffs(xi).isIdentity());
    EXPECT_EQ(field.eval_dvalue_dxi(xi).cols(), Dim);
    EXPECT_EQ(field.eval_dgrad_xi_dcoeffs(xi).rows(), 3 * Dim);
    EXPECT_TRUE(field.eval_dvalue_dxi(xi).isZero());
    EXPECT_TRUE(field.eval_dgrad_xi_dcoeffs(xi).isZero());
    EXPECT_EQ(field.eval_d2value_dxi2(xi).rows(), 3);
    EXPECT_EQ(field.eval_d2value_dxi2(xi).cols(), Dim * (Dim + 1) / 2);
    EXPECT_EQ(field.eval_dhess_xi_dcoeffs(xi).rows(), 3 * Dim * (Dim + 1) / 2);
    EXPECT_EQ(field.eval_dhess_xi_dcoeffs(xi).cols(), 3);
    EXPECT_TRUE(field.eval_d2value_dxi2(xi).isZero());
    EXPECT_TRUE(field.eval_dhess_xi_dcoeffs(xi).isZero());
    const Coordinate bad = Coordinate::Constant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(field.eval_value(bad), std::invalid_argument);
    EXPECT_THROW(field.eval_dvalue_dcoeffs(bad), std::invalid_argument);
    EXPECT_THROW(field.eval_dvalue_dxi(bad), std::invalid_argument);
    EXPECT_THROW(field.eval_dgrad_xi_dcoeffs(bad), std::invalid_argument);
    EXPECT_THROW(field.eval_d2value_dxi2(bad), std::invalid_argument);
    EXPECT_THROW(field.eval_dhess_xi_dcoeffs(bad), std::invalid_argument);
}
} // namespace

TEST(LocalField, ConstantFieldsSupportTwoAndThreeParameterDimensions) {
    check_constant_spatial_derivatives<2>();
    check_constant_spatial_derivatives<3>();
}
