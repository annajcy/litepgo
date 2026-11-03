#include "litepgo/mesh/self_intersections.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <array>
#include <cstddef>
#include <vector>

namespace litepgo::mesh {
bool has_self_intersections(const TriangleMesh& mesh) {
    mesh.validate();

    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    std::vector<Kernel::Point_3> points;
    points.reserve(mesh.topology.num_vertices());
    for (Eigen::Index offset = 0; offset < mesh.positions.size(); offset += 3) {
        points.emplace_back(mesh.positions[offset], mesh.positions[offset + 1], mesh.positions[offset + 2]);
    }

    std::vector<std::array<std::size_t, 3>> triangles;
    triangles.reserve(mesh.topology.num_cells());
    for (int cell = 0; cell < mesh.topology.num_cells(); ++cell) {
        const auto& vertices = mesh.topology.cell_vertices(cell);
        triangles.push_back({static_cast<std::size_t>(vertices[0]), static_cast<std::size_t>(vertices[1]),
                             static_cast<std::size_t>(vertices[2])});
    }

    return CGAL::Polygon_mesh_processing::does_triangle_soup_self_intersect(points, triangles);
}
} // namespace litepgo::mesh
