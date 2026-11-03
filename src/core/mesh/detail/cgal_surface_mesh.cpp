#include "litepgo/mesh/detail/cgal_surface_mesh.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace litepgo::mesh::detail {
CgalSurfaceMesh to_cgal_surface_mesh(const TriangleMesh& mesh) {
    mesh.validate();
    CgalSurfaceMesh surface;
    using Vertex = CgalSurfaceMesh::Vertex_index;
    for (int vertex = 0; vertex < mesh.topology.num_vertices(); ++vertex) {
        const Eigen::Index offset = 3 * Eigen::Index(vertex);
        surface.add_vertex({mesh.positions[offset], mesh.positions[offset + 1], mesh.positions[offset + 2]});
    }
    for (int face = 0; face < mesh.topology.num_cells(); ++face) {
        const auto& vertices = mesh.topology.cell_vertices(face);
        if (surface.add_face(Vertex(vertices[0]), Vertex(vertices[1]), Vertex(vertices[2])) ==
            CgalSurfaceMesh::null_face()) {
            throw std::invalid_argument("CGAL Surface_mesh cannot insert triangle " + std::to_string(face));
        }
    }
    return surface;
}

TriangleMesh from_cgal_surface_mesh(const CgalSurfaceMesh& surface) {
    const auto vertex_count = surface.number_of_vertices();
    const auto face_count = surface.number_of_faces();
    if (vertex_count > std::numeric_limits<int>::max() || face_count > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("CGAL Surface_mesh exceeds TriangleMesh index range");
    }
    Vec positions(3 * Eigen::Index(vertex_count));
    std::vector<int> vertex_ids(surface.num_vertices(), -1);
    int vertex_id = 0;
    for (const auto vertex : surface.vertices()) {
        const auto& point = surface.point(vertex);
        positions.segment<3>(3 * Eigen::Index(vertex_id)) << point.x(), point.y(), point.z();
        vertex_ids[vertex.idx()] = vertex_id++;
    }

    std::vector<TriangleTopology::Cell> triangles;
    triangles.reserve(face_count);
    for (const auto face : surface.faces()) {
        if (surface.degree(face) != 3) {
            throw std::invalid_argument("CGAL Surface_mesh contains a non-triangular face");
        }
        TriangleTopology::Cell triangle;
        int corner = 0;
        for (const auto vertex : surface.vertices_around_face(surface.halfedge(face))) {
            triangle[corner++] = vertex_ids[vertex.idx()];
        }
        triangles.push_back(triangle);
    }
    TriangleMesh mesh{std::move(positions), TriangleTopology(vertex_id, std::move(triangles))};
    mesh.validate();
    return mesh;
}
} // namespace litepgo::mesh::detail
