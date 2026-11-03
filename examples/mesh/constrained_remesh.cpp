#include "litepgo/mesh/isotropic_remesh.hpp"
#include "litepgo/mesh/self_intersections.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace {
using Point = std::array<double, 3>;
using Edge = std::pair<Point, Point>;

Point point(const litepgo::mesh::TriangleMesh& mesh, int vertex) {
    const Eigen::Index offset = 3 * Eigen::Index(vertex);
    return {mesh.positions[offset], mesh.positions[offset + 1], mesh.positions[offset + 2]};
}

std::set<Edge> boundary_segments(const litepgo::mesh::TriangleMesh& mesh) {
    std::map<std::pair<int, int>, int> incidence;
    for (int face = 0; face < mesh.topology.num_cells(); ++face) {
        const auto& triangle = mesh.topology.cell_vertices(face);
        for (int corner = 0; corner < 3; ++corner) {
            ++incidence[std::minmax(triangle[corner], triangle[(corner + 1) % 3])];
        }
    }
    std::set<Edge> boundary;
    for (const auto& [edge, count] : incidence) {
        if (count == 1) {
            const auto a = point(mesh, edge.first);
            const auto b = point(mesh, edge.second);
            boundary.insert(std::minmax(a, b));
        }
    }
    return boundary;
}
} // namespace

int main() {
    using namespace litepgo::mesh;
    litepgo::Vec positions(15);
    positions << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.2, 0.2, 0;
    const TriangleMesh input{positions, TriangleTopology(5, {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}})};
    const std::array fixed_vertices{4};
    const auto result = isotropic_remesh(input, 0.8, 5, fixed_vertices);

    int fixed_points = 0;
    for (int vertex = 0; vertex < result.topology.num_vertices(); ++vertex) {
        fixed_points += point(result, vertex) == point(input, fixed_vertices[0]);
    }
    if (fixed_points != 1 || boundary_segments(result) != boundary_segments(input) || has_self_intersections(result)) {
        throw std::runtime_error("Remeshed square failed its fixed point, boundary, or self-intersection check");
    }
    std::cout << input.topology.num_cells() << " -> " << result.topology.num_cells()
              << " triangles. Fixed point, boundary, and self-intersection checks passed.\n";
}
