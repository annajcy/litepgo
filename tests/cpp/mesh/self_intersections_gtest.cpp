#include "litepgo/mesh/self_intersections.hpp"

#include <gtest/gtest.h>

namespace litepgo::mesh {
namespace {
TEST(MeshSelfIntersections, AllowsSharedEdge) {
    Vec positions(12);
    positions << 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, -1, 0;
    EXPECT_FALSE(has_self_intersections({positions, TriangleTopology(4, {{0, 1, 2}, {1, 0, 3}})}));
}

TEST(MeshSelfIntersections, DetectsCrossing) {
    Vec positions(18);
    positions << 0, 0, 0, 2, 0, 0, 0, 2, 0, 0.5, 0.5, -1, 0.5, 0.5, 1, 0.5, 1.5, 0;
    EXPECT_TRUE(has_self_intersections({positions, TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})}));
}

TEST(MeshSelfIntersections, DetectsOverlapAcrossSharedEdge) {
    Vec positions(12);
    positions << 0, 0, 0, 2, 0, 0, 0, 2, 0, 0.5, 0.5, 0;
    EXPECT_TRUE(has_self_intersections({positions, TriangleTopology(4, {{0, 1, 2}, {1, 0, 3}})}));
}

TEST(MeshSelfIntersections, ReportsDegenerateTriangle) {
    Vec positions(9);
    positions << 0, 0, 0, 1, 0, 0, 2, 0, 0;
    EXPECT_TRUE(has_self_intersections({positions, TriangleTopology(3, {{0, 1, 2}})}));
}

TEST(MeshSelfIntersections, AllowsEmptyMesh) {
    EXPECT_FALSE(has_self_intersections({Vec(0), TriangleTopology(0, {})}));
}
} // namespace
} // namespace litepgo::mesh
