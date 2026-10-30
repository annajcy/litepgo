#include "litepgo/dof/surface_kinematics.hpp"
#include "litepgo/dof/vertex_displacement_map.hpp"
#include "litepgo/mesh/extract_surface.hpp"
#include <gtest/gtest.h>
using namespace litepgo;
TEST(VertexDisplacementMap, ExactExtractedBoundary) {
    Vec X(15);
    X << 0, 0, 0, .25, .25, .25, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    mesh::TetMesh volume{X, mesh::TetTopology(5, {{1, 2, 3, 4}, {0, 1, 3, 4}, {0, 2, 1, 4}, {0, 2, 3, 1}})};
    auto surface = mesh::extract_surface_with_vertices(volume);
    EXPECT_EQ(surface.vertices, (std::vector<int>{0, 2, 3, 4}));
    auto W = dof::build_vertex_displacement_map(surface.vertices, 15);
    EXPECT_EQ(W.nonZeros(), 12);
    const Vec u = Vec::LinSpaced(15, 0.01, 0.15);
    const dof::SurfaceKinematics kinematics(surface.mesh.positions, W);
    EXPECT_EQ(kinematics.positions(u), mesh::extract_surface(mesh::TetMesh{X + u, volume.topology}).positions);
}
TEST(VertexDisplacementMap, ValueSlotsAndInvalidInput) {
    auto W = dof::build_vertex_displacement_map({2, 0, 1}, 72, 24);
    EXPECT_EQ(W.nonZeros(), 9);
    Vec result = W * Vec::LinSpaced(72, 0, 71);
    EXPECT_EQ(result, (Vec(9) << 48, 49, 50, 0, 1, 2, 24, 25, 26).finished());
    EXPECT_THROW(dof::build_vertex_displacement_map({-1}, 3), std::out_of_range);
    EXPECT_THROW(dof::build_vertex_displacement_map({1}, 3), std::out_of_range);
    EXPECT_THROW(dof::build_vertex_displacement_map({}, -1), std::invalid_argument);
    EXPECT_THROW(dof::build_vertex_displacement_map({0}, 3, 2), std::invalid_argument);
    EXPECT_EQ(dof::build_vertex_displacement_map({}, 9).rows(), 0);
}
