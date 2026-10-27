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
    const IdentityField field(input.reshaped());
    input.setZero();
    Eigen::Vector3d first_point, second_point;
    first_point.setConstant(0.2);
    second_point.setConstant(100.0);
    EXPECT_EQ(field.value(first_point), field.value(second_point));
    EXPECT_EQ(field.value(Eigen::Vector3d::Zero())[8], 9);
    EXPECT_TRUE(field.dvalue_dcoeffs(Eigen::Vector3d::Zero()).isIdentity());
    const auto original = field.value(Eigen::Vector3d::Zero());
    for (int j = 0; j < 9; ++j) {
        auto perturbed = original;
        perturbed.data()[j] += 1e-4;
        const Eigen::VectorXd fd = (IdentityField(perturbed).value(Eigen::Vector3d::Zero()) - original) / 1e-4;
        EXPECT_LT((fd - field.dvalue_dcoeffs(Eigen::Vector3d::Zero()).col(j)).norm(), 1e-10);
    }
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> row_major = original.reshaped(3, 3);
    EXPECT_EQ(IdentityField(row_major.reshaped()).value(Eigen::Vector3d::Zero()), original);
    input(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((IdentityField(input.reshaped())), std::invalid_argument);
}

TEST(LocalField, ExplicitlyExtractedValuesOwnOnlySelectedSnapshots) {
    litepgo::Vec data = Vec::Zero(18);
    const ElementDofMap map(TetTopology(4, {{0, 1, 2, 3}, {0, 1, 2, 3}}), 9);
    data.head<9>() = Eigen::Matrix3d::Identity().eval().reshaped();
    data.tail<9>() = (2 * Eigen::Matrix3d::Identity()).eval().reshaped();
    const litepgo::IdentityField first(map.gather(0, data));
    const litepgo::IdentityField second(map.gather(1, data));
    EXPECT_EQ(first.num_dofs(), 9);
    EXPECT_EQ(data.size(), 18);
    data.head<9>() = (3 * Eigen::Matrix3d::Identity()).eval().reshaped();
    EXPECT_EQ(first.value(Eigen::Vector3d::Zero()), Eigen::Matrix3d::Identity().reshaped());
    EXPECT_EQ(second.value(Eigen::Vector3d::Zero()), (2 * Eigen::Matrix3d::Identity()).reshaped());
    EXPECT_THROW((litepgo::IdentityField(map.gather(-1, data))), std::out_of_range);
    EXPECT_THROW((litepgo::IdentityField(map.gather(2, data))), std::out_of_range);
    data[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField(map.gather(0, data))), std::invalid_argument);
    const auto owned = [] {
        litepgo::Vec temporary = Vec::Zero(2);
        temporary << 4, 5;
        return litepgo::IdentityField(ElementDofMap(TetTopology(4, {{0, 1, 2, 3}}), 2).gather(0, temporary));
    }();
    EXPECT_EQ(owned.value(Eigen::Vector3d::Zero()), Eigen::Vector2d(4, 5));

    litepgo::Vec shared(Eigen::Vector2d(2, 3));
    const litepgo::IdentityField snapshot(shared);
    shared << 4, 5;
    EXPECT_EQ(snapshot.value(Eigen::Vector3d::Zero()), Eigen::Vector2d(2, 3));
    EXPECT_EQ(snapshot.value(Eigen::Vector3d::Zero()), snapshot.value(Eigen::Vector3d::Zero()));
    EXPECT_TRUE(snapshot.dvalue_dcoeffs(Eigen::Vector3d::Zero()).isIdentity());
    const auto shared_owned = [] {
        const litepgo::Vec temporary(Eigen::Vector2d(6, 7));
        return litepgo::IdentityField(temporary);
    }();
    EXPECT_EQ(shared_owned.value(Eigen::Vector3d::Zero()), Eigen::Vector2d(6, 7));
    shared[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((litepgo::IdentityField(shared)), std::invalid_argument);
}

TEST(LocalField, FlattenedMatrixJacobianUsesColumnMajorDofs) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> input;
    input << 1, 2, 3, 4;
    const litepgo::Vec data(input.reshaped().eval());
    const litepgo::IdentityField field(data);
    EXPECT_EQ(field.value(Eigen::Vector3d::Zero()), Eigen::Vector4d(1, 3, 2, 4));
    for (int j = 0; j < 4; ++j) {
        auto plus = data, minus = data;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::VectorXd fd = (litepgo::IdentityField(plus).value(Eigen::Vector3d::Zero()) -
                                    litepgo::IdentityField(minus).value(Eigen::Vector3d::Zero())) /
                                   2e-4;
        EXPECT_LT((fd - field.dvalue_dcoeffs(Eigen::Vector3d::Zero()).col(j)).norm(), 1e-10);
    }
}

TEST(LocalField, IdentityFieldReturnsDynamicVectorsAndValidatesValues) {
    const Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::VectorXd input = Eigen::VectorXd::LinSpaced(4, 1, 4);
    const litepgo::IdentityField field(input);
    static_assert(std::is_same_v<decltype(field.value(point)), Eigen::VectorXd>);
    const litepgo::IdentityField fixed(Eigen::Vector2d(2, 3));
    static_assert(std::is_same_v<decltype(fixed.value(point)), Eigen::VectorXd>);
    EXPECT_EQ(field.num_dofs(), 4);
    EXPECT_TRUE(field.dvalue_dcoeffs(point).isIdentity());
    input.setZero();
    EXPECT_EQ(field.value(point), Eigen::VectorXd::LinSpaced(4, 1, 4));
    for (int j = 0; j < 4; ++j) {
        auto plus = field.value(point), minus = plus;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::VectorXd fd =
            (litepgo::IdentityField(plus).value(point) - litepgo::IdentityField(minus).value(point)) / 2e-4;
        EXPECT_TRUE(fd.isApprox(field.dvalue_dcoeffs(point).col(j), 1e-10));
    }
    EXPECT_THROW((litepgo::IdentityField{Eigen::VectorXd{}}), std::invalid_argument);
    input[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField{input}), std::invalid_argument);
    const Eigen::Vector3d invalid_xi = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(field.value(invalid_xi), std::invalid_argument);
    EXPECT_THROW(field.dvalue_dxi(invalid_xi), std::invalid_argument);
    EXPECT_THROW(field.dvalue_dcoeffs(invalid_xi), std::invalid_argument);
    EXPECT_THROW(field.dgrad_xi_dcoeffs(invalid_xi), std::invalid_argument);
}
