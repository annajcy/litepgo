#include "litepgo/fem/dof/element_dof_data.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <utility>
namespace {
using namespace litepgo;
template <int N>
concept CanStore = requires { typename ElementDofData<N>; };
static_assert(CanStore<9> && CanStore<1> && !CanStore<0> && !CanStore<-1>);
static_assert(ElementDofData<9>::dofs_per_element() == 9);
TEST(ElementDofData, OwnsBlocksAndViews) {
    Vec source = Vec::LinSpaced(20, 0, 19);
    ElementDofData<9> data(source.segment(1, 18));
    EXPECT_EQ(data.num_elements(), 2);
    EXPECT_EQ(data.num_dofs(), 18);
    source.setZero();
    for (int e = 0; e < 2; ++e)
        for (int j = 0; j < 9; ++j)
            EXPECT_EQ(data(e, j), 9 * e + j + 1);
    data(1, 2) = 99;
    EXPECT_EQ(data.values()[11], 99);
    data.values()[1] = 42;
    const auto& view = data;
    EXPECT_EQ(view(0, 1), 42);
    EXPECT_EQ(view.values()[11], 99);
    auto copy = data;
    copy.values().setZero();
    EXPECT_EQ(data(1, 2), 99);
    auto moved = std::move(copy);
    EXPECT_EQ(moved.num_elements(), 2);
    copy = ElementDofData<9>(1);
    EXPECT_TRUE(copy.values().isZero());
}
TEST(ElementDofData, ValidatesCountsSizesAndSlots) {
    ElementDofData<3> data(2);
    EXPECT_TRUE(data.values().isZero());
    EXPECT_THROW(ElementDofData<3>(-1), std::invalid_argument);
    EXPECT_THROW((ElementDofData<3>(std::numeric_limits<int>::max())), std::invalid_argument);
    EXPECT_THROW(ElementDofData<3>(Vec::Zero(5)), std::invalid_argument);
    for (int e : {-1, 2}) {
        EXPECT_THROW(data(e, 0), std::out_of_range);
        EXPECT_THROW(std::as_const(data)(e, 0), std::out_of_range);
    }
    for (int j : {-1, 3}) {
        EXPECT_THROW(data(0, j), std::out_of_range);
        EXPECT_THROW(std::as_const(data)(0, j), std::out_of_range);
    }
    const ElementDofData<3> empty(0), copied_empty(Vec::Zero(0));
    EXPECT_EQ(empty.num_elements(), 0);
    EXPECT_EQ(copied_empty.num_dofs(), 0);
    EXPECT_THROW(empty(0, 0), std::out_of_range);
}
} // namespace
