#include "litepgo/fem/vertex_dof_layout.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

using litepgo::VertexDofLayout;

template <int Width>
concept CanFormLayout = requires { typename VertexDofLayout<Width>; };

static_assert(CanFormLayout<1>);
static_assert(CanFormLayout<3>);
static_assert(!CanFormLayout<0>);
static_assert(!CanFormLayout<-1>);
static_assert(std::is_constructible_v<VertexDofLayout<3>, int>);
static_assert(!std::is_convertible_v<int, VertexDofLayout<3>>);
static_assert(!std::is_polymorphic_v<VertexDofLayout<3>>);
static_assert(std::is_trivially_copyable_v<VertexDofLayout<3>>);
static_assert(VertexDofLayout<3>::dofs_per_vertex() == 3);

} // namespace

TEST(VertexDofLayout, NumbersConsecutiveVertexBlocks) {
    const VertexDofLayout<6> layout(3);
    EXPECT_EQ(layout.num_vertices(), 3);
    EXPECT_EQ(layout.dofs_per_vertex(), 6);
    EXPECT_EQ(layout.num_dofs(), 18);
    int expected = 0;
    for (int vertex = 0; vertex < 3; ++vertex) {
        for (int slot = 0; slot < 6; ++slot) {
            EXPECT_EQ(layout.dof(vertex, slot), expected++);
        }
    }
}

TEST(VertexDofLayout, AllowsEmptyLayouts) {
    const VertexDofLayout<3> layout(0);
    EXPECT_EQ(layout.num_vertices(), 0);
    EXPECT_EQ(layout.num_dofs(), 0);
    EXPECT_THROW(layout.dof(0, 0), std::out_of_range);
}

TEST(VertexDofLayout, RejectsNegativeCountsAndOverflow) {
    EXPECT_THROW(VertexDofLayout<3>(-1), std::invalid_argument);
    constexpr int largest_count = std::numeric_limits<int>::max() / 3;
    const VertexDofLayout<3> layout(largest_count);
    EXPECT_EQ(layout.num_dofs(), 3 * largest_count);
    EXPECT_EQ(layout.dof(largest_count - 1, 2), layout.num_dofs() - 1);
    EXPECT_THROW(VertexDofLayout<3>(largest_count + 1), std::invalid_argument);
}

TEST(VertexDofLayout, SupportsIntBoundaryWithoutAllocatingPerVertexStorage) {
    constexpr int max = std::numeric_limits<int>::max();
    const VertexDofLayout<1> many_vertices(max);
    EXPECT_EQ(many_vertices.num_dofs(), max);
    EXPECT_EQ(many_vertices.dof(max - 1, 0), max - 1);

    const VertexDofLayout<max> wide_vertex(1);
    EXPECT_EQ(wide_vertex.num_dofs(), max);
    EXPECT_EQ(wide_vertex.dof(0, max - 1), max - 1);
    EXPECT_THROW(VertexDofLayout<max>(2), std::invalid_argument);
}

TEST(VertexDofLayout, ChecksBothIndices) {
    const VertexDofLayout<3> layout(2);
    EXPECT_THROW(layout.dof(-1, 0), std::out_of_range);
    EXPECT_THROW(layout.dof(2, 0), std::out_of_range);
    EXPECT_THROW(layout.dof(0, -1), std::out_of_range);
    EXPECT_THROW(layout.dof(0, 3), std::out_of_range);
}
