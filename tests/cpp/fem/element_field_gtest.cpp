#include "litepgo/fem/element_field.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <utility>

namespace {
using namespace litepgo;
using Matrix = Eigen::Matrix3d;
template <class V>
concept CanStore = requires { typename ElementField<V>; };
static_assert(CanStore<Matrix> && CanStore<Eigen::Vector2d>);
static_assert(!CanStore<double> && !CanStore<Eigen::MatrixXd> && !CanStore<Eigen::Matrix3f>);
static_assert(!CanStore<Eigen::Matrix<double, 0, 1>>);

TEST(ElementField, OwnsColumnMajorBlocksAndViews) {
    std::array<Matrix, 2> source;
    source[0] << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    source[1] = 2 * source[0];
    ElementField<Matrix> field{std::span<const Matrix>(source)};
    EXPECT_EQ(field.num_elements(), 2);
    EXPECT_EQ(field.num_dofs(), 18);
    for (int e = 0; e < 2; ++e) {
        EXPECT_EQ(field.value(e), source[static_cast<std::size_t>(e)]);
        const auto indices = field.dof_indices(e);
        for (int j = 0; j < 9; ++j) {
            EXPECT_EQ(indices[static_cast<std::size_t>(j)], 9 * e + j);
            EXPECT_DOUBLE_EQ(field.values()[9 * e + j], source[static_cast<std::size_t>(e)](j % 3, j / 3));
        }
    }
    source[0].setZero();
    EXPECT_DOUBLE_EQ(field.value(0)(0, 0), 1);
    field.values()[1] = 99;
    EXPECT_DOUBLE_EQ(field.value(0)(1, 0), 99);
    Matrix snapshot = field.value(0);
    snapshot.setZero();
    EXPECT_DOUBLE_EQ(field.value(0)(1, 0), 99);
    field.set_value(1, Matrix::Identity());
    EXPECT_TRUE(field.value(1).isIdentity());
    auto copy = field;
    copy.values().setZero();
    EXPECT_FALSE(field.value(1).isZero());
    auto moved = std::move(copy);
    EXPECT_EQ(moved.num_elements(), 2);
    copy = ElementField<Matrix>(1);
    EXPECT_EQ(copy.num_dofs(), 9);
    const auto& const_field = field;
    EXPECT_EQ(const_field.values()[1], 99);
}

TEST(ElementField, StorageOrderAndBounds) {
    using RowMatrix = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
    ElementField<RowMatrix> field(1);
    EXPECT_TRUE(field.value(0).isZero());
    RowMatrix value;
    value << 1, 2, 3, 4;
    field.set_value(0, value);
    EXPECT_EQ(field.value(0), value);
    EXPECT_DOUBLE_EQ(field.values()[1], 3);
    EXPECT_THROW(ElementField<Matrix>(-1), std::invalid_argument);
    EXPECT_THROW((ElementField<Matrix>(std::numeric_limits<int>::max())), std::invalid_argument);
    for (int e : {-1, 1}) {
        EXPECT_THROW(field.value(e), std::out_of_range);
        EXPECT_THROW(field.set_value(e, value), std::out_of_range);
        EXPECT_THROW(field.dof_indices(e), std::out_of_range);
    }
    const ElementField<Matrix> empty(0);
    EXPECT_EQ(empty.num_elements(), 0);
    EXPECT_THROW(empty.value(0), std::out_of_range);
    const ElementField<Matrix> from_span{std::span<const Matrix>{}};
    EXPECT_EQ(from_span.num_dofs(), 0);
}
} // namespace
