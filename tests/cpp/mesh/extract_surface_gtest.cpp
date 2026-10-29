#include "litepgo/mesh/extract_surface.hpp"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <array>
#include <map>
#include <stdexcept>
#include <utility>

namespace litepgo::mesh::test {

namespace {

// Volume enclosed by outward-wound triangles, by the divergence theorem.
double enclosed_volume(const TriangleMesh& surface) {
    double volume = 0;
    for (int f = 0; f < surface.topology.num_cells(); ++f) {
        const auto& face = surface.topology.cell_vertices(f);
        const Eigen::Vector3d a = surface.positions.segment<3>(3 * face[0]);
        const Eigen::Vector3d b = surface.positions.segment<3>(3 * face[1]);
        const Eigen::Vector3d c = surface.positions.segment<3>(3 * face[2]);
        volume += a.dot(b.cross(c)) / 6;
    }
    return volume;
}

// Each directed edge appears once and its reverse once: closed and consistently wound.
bool closed_and_consistent(const TriangleMesh& surface) {
    std::map<std::pair<int, int>, int> edges;
    for (int f = 0; f < surface.topology.num_cells(); ++f) {
        const auto& face = surface.topology.cell_vertices(f);
        for (int j = 0; j < 3; ++j)
            ++edges[{face[j], face[(j + 1) % 3]}];
    }
    for (const auto& [edge, count] : edges) {
        const auto reverse = edges.find({edge.second, edge.first});
        if (count != 1 || reverse == edges.end() || reverse->second != 1)
            return false;
    }
    return true;
}

Vec unit_cube_corners(double scale, const Eigen::Vector3d& origin) {
    Eigen::Matrix<double, 8, 3, Eigen::RowMajor> corners;
    corners << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    corners = (scale * corners).rowwise() + origin.transpose();
    return corners.reshaped<Eigen::RowMajor>();
}

} // namespace

TEST(ExtractSurface, SingleTetDropsUnusedVertices) {
    Vec positions(15);
    positions << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 5, 5;
    const TetMesh mesh{positions, TetTopology(5, {{0, 1, 2, 3}})};
    const TriangleMesh surface = litepgo::mesh::extract_surface(mesh);
    EXPECT_EQ(surface.topology.num_vertices(), 4);
    EXPECT_EQ(surface.topology.num_cells(), 4);
    EXPECT_EQ(surface.positions, positions.head(12));
    EXPECT_TRUE(closed_and_consistent(surface));
    EXPECT_NEAR(enclosed_volume(surface), 1.0 / 6, 1e-15);
}

TEST(ExtractSurface, TwoHexesShareAnInteriorFace) {
    // Boxes [0,2]^3 and [2,4]x[0,2]^2 share the face x = 2.
    Vec positions(36);
    positions << unit_cube_corners(2, Eigen::Vector3d::Zero()), Vec::Zero(12);
    const Vec right = unit_cube_corners(2, Eigen::Vector3d(2, 0, 0));
    for (int j = 0; j < 4; ++j) // right corners 1, 2, 5, 6 are new vertices 8..11
        positions.segment<3>(3 * (8 + j)) = right.segment<3>(3 * std::array{1, 2, 5, 6}[j]);
    const HexMesh mesh{positions, HexTopology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}})};
    const TriangleMesh surface = litepgo::mesh::extract_surface(litepgo::mesh::VolumeMesh(mesh));
    EXPECT_EQ(surface.topology.num_vertices(), 12);
    EXPECT_EQ(surface.topology.num_cells(), 20); // 10 boundary quads
    EXPECT_TRUE(closed_and_consistent(surface));
    EXPECT_NEAR(enclosed_volume(surface), 16, 1e-13);
}

TEST(ExtractSurface, RejectsFaceSharedByThreeCells) {
    Vec positions(18);
    positions << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 1, 1, 1;
    const TetMesh mesh{positions, TetTopology(6, {{0, 1, 2, 3}, {0, 2, 1, 4}, {0, 1, 2, 5}})};
    EXPECT_THROW(litepgo::mesh::extract_surface(mesh), std::invalid_argument);
}

} // namespace litepgo::mesh::test
