#include "litepgo/fem/constant_field.hpp"
#include <gtest/gtest.h>

namespace {
using namespace litepgo;
template <class V>
concept CanStore = requires { typename ConstantField<V>; };
static_assert(CanStore<Eigen::Vector2d> && CanStore<Eigen::Matrix3d>);
static_assert(!CanStore<Eigen::VectorXd> && !CanStore<Eigen::Vector2f> && !CanStore<Eigen::Matrix<double, 0, 1>>);

TEST(ConstantField, SharedIndicesAndIndependentSnapshot) {
    Eigen::Vector2d theta(2, 3);
    ConstantField<Eigen::Vector2d> field(theta);
    theta.setZero();
    EXPECT_EQ(field.num_dofs, 2);
    const std::array<int, 2> expected{0, 1};
    EXPECT_EQ(field.dof_indices(), expected);
    EXPECT_EQ(field.value(), Eigen::Vector2d(2, 3));
    field.values()[0] = 5;
    EXPECT_EQ(field.value(), Eigen::Vector2d(5, 3));
    field.set_value({7, 8});
    const auto& const_field = field;
    EXPECT_EQ(const_field.values()[1], 8);
    auto copy = field;
    copy.values().setZero();
    EXPECT_EQ(field.value(), Eigen::Vector2d(7, 8));
}
TEST(ConstantField, MatrixValuesUseColumnMajorDofs) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> matrix;
    matrix << 1, 2, 3, 4;
    ConstantField<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> field(matrix);
    EXPECT_EQ(field.value(), matrix);
    EXPECT_DOUBLE_EQ(field.values()[1], 3);
    field.values()[2] = 9;
    EXPECT_DOUBLE_EQ(field.value()(0, 1), 9);
}
} // namespace
