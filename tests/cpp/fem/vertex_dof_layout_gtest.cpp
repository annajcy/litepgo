#include "litepgo/fem/vertex_dof_layout.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {

using litepgo::VertexDofLayout;

class TestLayout final : public VertexDofLayout {
  public:
    TestLayout(int num_vertices, int dofs_per_vertex) : VertexDofLayout(num_vertices, dofs_per_vertex) {}
};

static_assert(!std::is_constructible_v<VertexDofLayout, int, int>);
static_assert(std::has_virtual_destructor_v<VertexDofLayout>);

} // namespace

TEST(VertexDofLayout, NumbersConsecutiveVertexBlocks) {
    const TestLayout layout(3, 6);
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
    const TestLayout layout(0, 6);
    EXPECT_EQ(layout.num_vertices(), 0);
    EXPECT_EQ(layout.dofs_per_vertex(), 6);
    EXPECT_EQ(layout.num_dofs(), 0);
    EXPECT_THROW(layout.dof(0, 0), std::out_of_range);
}

TEST(VertexDofLayout, RejectsInvalidCountsAndOverflow) {
    EXPECT_THROW(TestLayout(-1, 3), std::invalid_argument);
    EXPECT_THROW(TestLayout(1, 0), std::invalid_argument);
    EXPECT_THROW(TestLayout(0, 0), std::invalid_argument);
    EXPECT_THROW(TestLayout(1, -1), std::invalid_argument);
    EXPECT_THROW(TestLayout(std::numeric_limits<int>::max(), 2), std::invalid_argument);
}

TEST(VertexDofLayout, SupportsIntBoundaryWithoutAllocatingPerVertexStorage) {
    constexpr int max = std::numeric_limits<int>::max();
    const TestLayout many_vertices(max, 1);
    EXPECT_EQ(many_vertices.num_dofs(), max);
    EXPECT_EQ(many_vertices.dof(max - 1, 0), max - 1);

    const TestLayout wide_vertex(1, max);
    EXPECT_EQ(wide_vertex.num_dofs(), max);
    EXPECT_EQ(wide_vertex.dof(0, max - 1), max - 1);
}

TEST(VertexDofLayout, ChecksBothIndices) {
    const TestLayout layout(2, 6);
    EXPECT_THROW(layout.dof(-1, 0), std::out_of_range);
    EXPECT_THROW(layout.dof(2, 0), std::out_of_range);
    EXPECT_THROW(layout.dof(0, -1), std::out_of_range);
    EXPECT_THROW(layout.dof(0, 6), std::out_of_range);
}

TEST(VertexDofLayout, DestroysSubclassThroughBasePointer) {
    class TrackedLayout final : public VertexDofLayout {
      public:
        explicit TrackedLayout(bool& destroyed) : VertexDofLayout(1, 3), destroyed_(destroyed) {}
        ~TrackedLayout() override {
            destroyed_ = true;
        }

      private:
        bool& destroyed_;
    };

    bool destroyed = false;
    {
        const std::unique_ptr<VertexDofLayout> layout = std::make_unique<TrackedLayout>(destroyed);
        EXPECT_EQ(layout->num_dofs(), 3);
        EXPECT_EQ(layout->dof(0, 2), 2);
    }
    EXPECT_TRUE(destroyed);
}
