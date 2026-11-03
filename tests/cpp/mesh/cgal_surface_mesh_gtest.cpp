#include "litepgo/mesh/detail/cgal_surface_mesh.hpp"

#include <CGAL/boost/graph/Euler_operations.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

namespace litepgo::mesh::detail {
namespace {
TriangleTopology::Cell canonical_winding(TriangleTopology::Cell triangle) {
    std::rotate(triangle.begin(), std::min_element(triangle.begin(), triangle.end()), triangle.end());
    return triangle;
}

TEST(CgalSurfaceMesh, RoundTripPreservesCoordinatesAndWinding) {
    Vec positions(15);
    positions << 0, 0, 0.125, 1, 0, 0.25, 1, 1, 0.5, 0, 1, 0.75, 9, 8, 7;
    const TriangleMesh input{positions, TriangleTopology(5, {{0, 1, 2}, {0, 2, 3}})};

    const auto surface = to_cgal_surface_mesh(input);
    for (const auto vertex : surface.vertices()) {
        const auto offset = 3 * Eigen::Index(vertex.idx());
        EXPECT_EQ(surface.point(vertex),
                  CgalSurfaceMesh::Point(positions[offset], positions[offset + 1], positions[offset + 2]));
    }
    const auto output = from_cgal_surface_mesh(surface);
    EXPECT_EQ(output.positions, positions);
    ASSERT_EQ(output.topology.num_cells(), 2);
    for (int face = 0; face < output.topology.num_cells(); ++face) {
        EXPECT_EQ(canonical_winding(output.topology.cell_vertices(face)), input.topology.cell_vertices(face));
    }
}

TEST(CgalSurfaceMesh, CompactsLiveVerticesAndFacesWithoutCollectingGarbage) {
    Vec positions(18);
    positions << 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 2, 1, 0;
    auto surface = to_cgal_surface_mesh({positions, TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})});
    CGAL::Euler::remove_face(surface.halfedge(CgalSurfaceMesh::Face_index(0)), surface);
    ASSERT_TRUE(surface.has_garbage());
    ASSERT_EQ(surface.number_of_vertices(), 3);

    const auto output = from_cgal_surface_mesh(surface);
    EXPECT_EQ(output.positions, positions.tail(9));
    ASSERT_EQ(output.topology.num_cells(), 1);
    EXPECT_EQ(canonical_winding(output.topology.cell_vertices(0)), (TriangleTopology::Cell{0, 1, 2}));
    EXPECT_TRUE(surface.has_garbage());
}

TEST(CgalSurfaceMesh, RejectsIncompatibleFaceWinding) {
    Vec positions(12);
    positions << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0;
    const TriangleMesh input{positions, TriangleTopology(4, {{0, 1, 2}, {0, 1, 3}})};
    EXPECT_THROW(to_cgal_surface_mesh(input), std::invalid_argument);
}

TEST(CgalSurfaceMesh, RejectsNonTriangularFace) {
    CgalSurfaceMesh surface;
    const auto a = surface.add_vertex({0, 0, 0});
    const auto b = surface.add_vertex({1, 0, 0});
    const auto c = surface.add_vertex({1, 1, 0});
    const auto d = surface.add_vertex({0, 1, 0});
    ASSERT_NE(surface.add_face(a, b, c, d), CgalSurfaceMesh::null_face());
    EXPECT_THROW(from_cgal_surface_mesh(surface), std::invalid_argument);
}

TEST(CgalSurfaceMesh, ValidatesCoordinatesInBothDirections) {
    Vec positions(3);
    positions << std::numeric_limits<double>::infinity(), 0, 0;
    EXPECT_THROW(to_cgal_surface_mesh((TriangleMesh{positions, TriangleTopology(1, {})})), std::invalid_argument);

    CgalSurfaceMesh surface;
    surface.add_vertex({positions[0], 0, 0});
    EXPECT_THROW(from_cgal_surface_mesh(surface), std::invalid_argument);
}
} // namespace
} // namespace litepgo::mesh::detail
