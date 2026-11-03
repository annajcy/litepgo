#include "litepgo/mesh/detail/cgal_surface_mesh.hpp"
#include "litepgo/mesh/isotropic_remesh.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <utility>

namespace litepgo::mesh {
namespace {
TriangleMesh square_fan() {
    Vec positions(15);
    positions << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.2, 0.2, 0;
    return {positions, TriangleTopology(5, {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}})};
}

using Point = std::array<double, 3>;
std::set<std::pair<Point, Point>> boundary_segments(const TriangleMesh& mesh) {
    const auto surface = detail::to_cgal_surface_mesh(mesh);
    std::set<std::pair<Point, Point>> segments;
    for (const auto edge : surface.edges()) {
        if (surface.is_border(edge)) {
            const auto halfedge = surface.halfedge(edge);
            const auto& source = surface.point(surface.source(halfedge));
            const auto& target = surface.point(surface.target(halfedge));
            const Point a{source.x(), source.y(), source.z()};
            const Point b{target.x(), target.y(), target.z()};
            segments.insert(std::minmax(a, b));
        }
    }
    return segments;
}

TEST(IsotropicRemesh, PreservesBoundarySegmentsAndFixedInteriorPoint) {
    const auto input = square_fan();
    const std::array fixed{4};
    const auto result = isotropic_remesh(input, 0.8, 5, fixed);
    const auto expected_boundary = boundary_segments(input);
    ASSERT_EQ(expected_boundary.size(), 4);
    EXPECT_EQ(boundary_segments(result), expected_boundary);
    EXPECT_NE(result.topology.num_cells(), input.topology.num_cells());

    int fixed_points = 0;
    for (Eigen::Index offset = 0; offset < result.positions.size(); offset += 3) {
        fixed_points += result.positions.segment<3>(offset) == input.positions.tail<3>();
    }
    EXPECT_EQ(fixed_points, 1);
}

TEST(IsotropicRemesh, RejectsOverlongBoundary) {
    EXPECT_THROW(isotropic_remesh(square_fan(), 0.5, 2), std::invalid_argument);
}

TEST(IsotropicRemesh, RejectsInvalidArguments) {
    const auto input = square_fan();
    const std::array invalid_vertex{5};
    EXPECT_THROW(isotropic_remesh(input, 0.8, 2, invalid_vertex), std::invalid_argument);
    EXPECT_THROW(isotropic_remesh(input, 0.8, 0), std::invalid_argument);
    EXPECT_THROW(isotropic_remesh(input, std::numeric_limits<double>::infinity(), 2), std::invalid_argument);
}

TEST(IsotropicRemesh, RejectsDegenerateTriangle) {
    Vec positions(9);
    positions << 0, 0, 0, 1, 0, 0, 2, 0, 0;
    const TriangleMesh input{positions, TriangleTopology(3, {{0, 1, 2}})};
    EXPECT_THROW(isotropic_remesh(input, 2.0, 1), std::invalid_argument);
}
} // namespace
} // namespace litepgo::mesh
