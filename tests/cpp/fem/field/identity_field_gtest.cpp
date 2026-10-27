#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include <gtest/gtest.h>
#include <limits>

using namespace litepgo;

TEST(LocalField, DirectValuesIgnoreLocationAndOwnData) {
    Eigen::Matrix3d input;
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    const IdentityField<Eigen::Matrix3d> field(input);
    input.setZero();
    Eigen::Vector3d first_point, second_point;
    first_point.setConstant(0.2);
    second_point.setConstant(100.0);
    EXPECT_EQ(field.value(first_point), field.value(second_point));
    EXPECT_EQ(field.value(Eigen::Vector3d::Zero())(2, 2), 9);
    EXPECT_TRUE(field.evaluate(Eigen::Vector3d::Zero()).dvalue_dcoeffs().isIdentity());
    const auto original = field.value(Eigen::Vector3d::Zero());
    for (int j = 0; j < 9; ++j) {
        auto perturbed = original;
        perturbed.data()[j] += 1e-4;
        const Eigen::Matrix3d fd =
            (IdentityField<Eigen::Matrix3d>(perturbed).value(Eigen::Vector3d::Zero()) - original) / 1e-4;
        EXPECT_LT((Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fd.data()) -
                   field.evaluate(Eigen::Vector3d::Zero()).dvalue_dcoeffs().col(j))
                      .norm(),
                  1e-10);
    }
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> row_major = original;
    EXPECT_EQ((IdentityField<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(row_major).value(Eigen::Vector3d::Zero())),
              original);
    input(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((IdentityField<Eigen::Matrix3d>(input)), std::invalid_argument);
}

TEST(LocalField, ExplicitlyExtractedValuesOwnOnlySelectedSnapshots) {
    litepgo::Vec data = Vec::Zero(18);
    const ElementDofMap map(TetTopology(4, {{0, 1, 2, 3}, {0, 1, 2, 3}}), 9);
    data.head<9>() = Eigen::Matrix3d::Identity().eval().reshaped();
    data.tail<9>() = (2 * Eigen::Matrix3d::Identity()).eval().reshaped();
    const litepgo::IdentityField<Eigen::Matrix3d> first(map.gather(0, data).reshaped(3, 3));
    const litepgo::IdentityField<Eigen::Matrix3d> second(map.gather(1, data).reshaped(3, 3));
    EXPECT_EQ(first.num_dofs(), 9);
    EXPECT_EQ(data.size(), 18);
    data.head<9>() = (3 * Eigen::Matrix3d::Identity()).eval().reshaped();
    EXPECT_EQ(first.value(Eigen::Vector3d::Zero()), Eigen::Matrix3d::Identity());
    EXPECT_EQ(second.value(Eigen::Vector3d::Zero()), 2 * Eigen::Matrix3d::Identity());
    EXPECT_THROW((litepgo::IdentityField<Eigen::Matrix3d>(map.gather(-1, data).reshaped(3, 3))), std::out_of_range);
    EXPECT_THROW((litepgo::IdentityField<Eigen::Matrix3d>(map.gather(2, data).reshaped(3, 3))), std::out_of_range);
    data[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField<Eigen::Matrix3d>(map.gather(0, data).reshaped(3, 3))), std::invalid_argument);
    const auto owned = [] {
        litepgo::Vec temporary = Vec::Zero(2);
        temporary << 4, 5;
        return litepgo::IdentityField<Eigen::Vector2d>(
            ElementDofMap(TetTopology(4, {{0, 1, 2, 3}}), 2).gather(0, temporary));
    }();
    EXPECT_EQ(owned.value(Eigen::Vector3d::Zero()), Eigen::Vector2d(4, 5));

    litepgo::Vec shared(Eigen::Vector2d(2, 3));
    const litepgo::IdentityField<Eigen::Vector2d> snapshot(shared);
    shared << 4, 5;
    EXPECT_EQ(snapshot.value(Eigen::Vector3d::Zero()), Eigen::Vector2d(2, 3));
    EXPECT_EQ(snapshot.value(Eigen::Vector3d::Zero()), snapshot.value(Eigen::Vector3d::Zero()));
    EXPECT_TRUE(snapshot.evaluate(Eigen::Vector3d::Zero()).dvalue_dcoeffs().isIdentity());
    const auto shared_owned = [] {
        const litepgo::Vec temporary(Eigen::Vector2d(6, 7));
        return litepgo::IdentityField<Eigen::Vector2d>(temporary);
    }();
    EXPECT_EQ(shared_owned.value(Eigen::Vector3d::Zero()), Eigen::Vector2d(6, 7));
    shared[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((litepgo::IdentityField<Eigen::Vector2d>(shared)), std::invalid_argument);
}

TEST(LocalField, ConstantMatrixJacobianUsesColumnMajorDofs) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> input;
    input << 1, 2, 3, 4;
    const litepgo::Vec data(input.reshaped().eval());
    const litepgo::IdentityField<Eigen::Matrix2d> field(data.reshaped(2, 2));
    EXPECT_EQ(field.value(Eigen::Vector3d::Zero()), input);
    for (int j = 0; j < 4; ++j) {
        auto plus = data, minus = data;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::Matrix2d fd =
            (litepgo::IdentityField<Eigen::Matrix2d>(plus.reshaped(2, 2)).value(Eigen::Vector3d::Zero()) -
             litepgo::IdentityField<Eigen::Matrix2d>(minus.reshaped(2, 2)).value(Eigen::Vector3d::Zero())) /
            2e-4;
        EXPECT_LT((Eigen::Map<const Eigen::Vector4d>(fd.data()) -
                   field.evaluate(Eigen::Vector3d::Zero()).dvalue_dcoeffs().col(j))
                      .norm(),
                  1e-10);
    }
}

TEST(LocalField, IdentityFieldPreservesExplicitValueTypeAndValidatesDynamicValues) {
    const Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::VectorXd input = Eigen::VectorXd::LinSpaced(4, 1, 4);
    const litepgo::IdentityField<Eigen::VectorXd> field(input);
    static_assert(std::is_same_v<decltype(field.value(point)), Eigen::VectorXd>);
    const litepgo::IdentityField<Eigen::Vector2d> fixed(Eigen::Vector2d(2, 3));
    static_assert(std::is_same_v<decltype(fixed.value(point)), Eigen::Vector2d>);
    EXPECT_EQ(field.num_dofs(), 4);
    EXPECT_TRUE(field.evaluate(point).dvalue_dcoeffs().isIdentity());
    input.setZero();
    EXPECT_EQ(field.value(point), Eigen::VectorXd::LinSpaced(4, 1, 4));
    for (int j = 0; j < 4; ++j) {
        auto plus = field.value(point), minus = plus;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::VectorXd fd = (litepgo::IdentityField<Eigen::VectorXd>(plus).value(point) -
                                    litepgo::IdentityField<Eigen::VectorXd>(minus).value(point)) /
                                   2e-4;
        EXPECT_TRUE(fd.isApprox(field.evaluate(point).dvalue_dcoeffs().col(j), 1e-10));
    }
    EXPECT_THROW((litepgo::IdentityField<Eigen::VectorXd>{Eigen::VectorXd{}}), std::invalid_argument);
    input[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField<Eigen::VectorXd>{input}), std::invalid_argument);
}
