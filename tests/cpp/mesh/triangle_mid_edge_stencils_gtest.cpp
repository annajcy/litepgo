#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

namespace {
using namespace litepgo;

TEST(TriangleMidEdgeStencils, PreservesFaceOrderAndAppendsNeighborsInOppositeEdgeOrder) {
    const TriangleTopology topology(7, {{4, 1, 5}, {5, 1, 3}, {4, 5, 0}, {1, 4, 2}});
    const auto stencils = make_triangle_mid_edge_stencils(topology);
    const std::vector<std::vector<int>> expected{{4, 1, 5, 3, 0, 2}, {5, 1, 3, 4}, {4, 5, 0, 1}, {1, 4, 2, 5}};
    EXPECT_EQ(stencils.vertices, expected);
    ASSERT_EQ(stencils.opposite.size(), 4);
    EXPECT_EQ(stencils.opposite[0], (std::array<int, 3>{3, 4, 5}));
    for (std::size_t f = 1; f < stencils.opposite.size(); ++f)
        EXPECT_EQ(stencils.opposite[f], (std::array<int, 3>{-1, -1, 3}));
}

TEST(TriangleMidEdgeStencils, DeduplicatesSharedOppositeVerticesOnClosedMeshes) {
    const TriangleTopology topology(4, {{0, 1, 2}, {0, 3, 1}, {1, 3, 2}, {2, 3, 0}});
    const auto stencils = make_triangle_mid_edge_stencils(topology);
    const std::vector<std::vector<int>> expected{{0, 1, 2, 3}, {0, 3, 1, 2}, {1, 3, 2, 0}, {2, 3, 0, 1}};
    EXPECT_EQ(stencils.vertices, expected);
    for (const auto& opposite : stencils.opposite)
        EXPECT_EQ(opposite, (std::array<int, 3>{3, 3, 3}));
}

TEST(TriangleMidEdgeStencils, HandlesBoundariesDisconnectedFacesAndEmptyMeshes) {
    const auto stencils = make_triangle_mid_edge_stencils(TriangleTopology(7, {{3, 0, 4}, {1, 5, 2}}));
    EXPECT_EQ(stencils.vertices, (std::vector<std::vector<int>>{{3, 0, 4}, {1, 5, 2}}));
    for (const auto& opposite : stencils.opposite)
        EXPECT_EQ(opposite, (std::array<int, 3>{-1, -1, -1}));
    const auto empty = make_triangle_mid_edge_stencils(TriangleTopology(0, {}));
    EXPECT_TRUE(empty.vertices.empty());
    EXPECT_TRUE(empty.opposite.empty());
    EXPECT_TRUE(make_triangle_mid_edge_stencils(TriangleTopology(3, {})).vertices.empty());
}

TEST(TriangleMidEdgeStencils, RejectsDuplicateFacesInEitherOrientation) {
    EXPECT_THROW(make_triangle_mid_edge_stencils(TriangleTopology(3, {{0, 1, 2}, {1, 2, 0}})), std::invalid_argument);
    EXPECT_THROW(make_triangle_mid_edge_stencils(TriangleTopology(3, {{0, 1, 2}, {2, 1, 0}})), std::invalid_argument);
}

TEST(TriangleMidEdgeStencils, RejectsInconsistentWindingAndNonManifoldEdges) {
    EXPECT_THROW(make_triangle_mid_edge_stencils(TriangleTopology(4, {{0, 1, 2}, {1, 2, 3}})), std::invalid_argument);
    EXPECT_THROW(make_triangle_mid_edge_stencils(TriangleTopology(5, {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}})),
                 std::invalid_argument);
}
} // namespace
