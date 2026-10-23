#include "litepgo/fem/displacement_dof_layout.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <stdexcept>

using litepgo::DisplacementDofLayout;

TEST(DisplacementDofLayout, UsesVertexMajorXYZOrdering) {
    const std::shared_ptr<const litepgo::VertexDofLayout> base = std::make_shared<DisplacementDofLayout>(4);
    EXPECT_EQ(base->num_vertices(), 4);
    EXPECT_EQ(base->dofs_per_vertex(), 3);
    EXPECT_EQ(base->num_dofs(), 12);

    const DisplacementDofLayout layout(4);
    for (int vertex = 0; vertex < 4; ++vertex) {
        EXPECT_EQ(layout.displacement_dof(vertex, 0), 3 * vertex);
        EXPECT_EQ(layout.displacement_dof(vertex, 1), 3 * vertex + 1);
        EXPECT_EQ(layout.displacement_dof(vertex, 2), 3 * vertex + 2);
        EXPECT_EQ(base->dof(vertex, 2), layout.displacement_dof(vertex, 2));
    }
}

TEST(DisplacementDofLayout, AllowsEmptyLayouts) {
    const DisplacementDofLayout layout(0);
    EXPECT_EQ(layout.dofs_per_vertex(), 3);
    EXPECT_EQ(layout.num_dofs(), 0);
    EXPECT_THROW(layout.displacement_dof(0, 0), std::out_of_range);
}

TEST(DisplacementDofLayout, ChecksVerticesAndComponents) {
    const DisplacementDofLayout layout(2);
    EXPECT_THROW(layout.displacement_dof(-1, 0), std::out_of_range);
    EXPECT_THROW(layout.displacement_dof(2, 0), std::out_of_range);
    EXPECT_THROW(layout.displacement_dof(0, -1), std::out_of_range);
    EXPECT_THROW(layout.displacement_dof(0, 3), std::out_of_range);
}

TEST(DisplacementDofLayout, ChecksCountAndOverflowBoundary) {
    EXPECT_THROW(DisplacementDofLayout(-1), std::invalid_argument);
    constexpr int largest_count = std::numeric_limits<int>::max() / 3;
    const DisplacementDofLayout layout(largest_count);
    EXPECT_EQ(layout.num_dofs(), 3 * largest_count);
    EXPECT_EQ(layout.displacement_dof(largest_count - 1, 2), layout.num_dofs() - 1);
    EXPECT_THROW(DisplacementDofLayout(largest_count + 1), std::invalid_argument);
}
