#include "litepgo/fem/dof/constant_dof_data.hpp"
#include <gtest/gtest.h>
#include <utility>
namespace {
using namespace litepgo;
template <int N>
concept CanStore = requires { typename ConstantDofData<N>; };
static_assert(CanStore<2> && CanStore<1> && !CanStore<0> && !CanStore<-1>);
static_assert(ConstantDofData<2>::num_dofs() == 2);
TEST(ConstantDofData, OwnsSharedSlotsAndViews) {
    ConstantDofData<2> zero;
    EXPECT_TRUE(zero.values().isZero());
    Vec source = Vec::LinSpaced(4, 1, 4);
    ConstantDofData<2> data(source.segment(1, 2));
    source.setZero();
    EXPECT_EQ(data.values(), Eigen::Vector2d(2, 3));
    data(0) = 5;
    data.values()[1] = 8;
    const auto& view = data;
    EXPECT_EQ(view(0), 5);
    EXPECT_EQ(view.values()[1], 8);
    auto copy = data;
    copy.values().setZero();
    EXPECT_EQ(data.values(), Eigen::Vector2d(5, 8));
}
TEST(ConstantDofData, ValidatesSizeAndSlots) {
    for (int size : {0, 1, 3})
        EXPECT_THROW(ConstantDofData<2>(Vec::Zero(size)), std::invalid_argument);
    ConstantDofData<2> data;
    for (int slot : {-1, 2}) {
        EXPECT_THROW(data(slot), std::out_of_range);
        EXPECT_THROW(std::as_const(data)(slot), std::out_of_range);
    }
}
} // namespace
