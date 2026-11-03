#include "litepgo/mesh/isotropic_remesh.hpp"
#include "litepgo/mesh/detail/cgal_surface_mesh.hpp"

#include <CGAL/Polygon_mesh_processing/remesh.h>

#include <cmath>
#include <stdexcept>

namespace litepgo::mesh {
TriangleMesh isotropic_remesh(const TriangleMesh& mesh, double target_edge_length, int num_iter,
                              std::span<const int> constrained_vertices) {
    if (!std::isfinite(target_edge_length) || target_edge_length <= 0.0) {
        throw std::invalid_argument("target_edge_length must be finite and positive");
    }
    if (num_iter <= 0) {
        throw std::invalid_argument("num_iter must be positive");
    }

    auto surface = detail::to_cgal_surface_mesh(mesh);
    using Surface = detail::CgalSurfaceMesh;
    using Vertex = Surface::Vertex_index;
    auto fixed = surface.add_property_map<Vertex, bool>("v:fixed", false).first;
    for (const int vertex : constrained_vertices) {
        if (vertex < 0 || vertex >= mesh.topology.num_vertices()) {
            throw std::invalid_argument("constrained vertex index out of range");
        }
        fixed[Vertex(vertex)] = true;
    }
    if (surface.number_of_faces() == 0) {
        return mesh;
    }
    for (int cell = 0; cell < mesh.topology.num_cells(); ++cell) {
        const auto& ids = mesh.topology.cell_vertices(cell);
        if (CGAL::collinear(surface.point(Vertex(ids[0])), surface.point(Vertex(ids[1])),
                            surface.point(Vertex(ids[2])))) {
            throw std::invalid_argument("isotropic remeshing requires nondegenerate triangles");
        }
    }

    const double maximum_length = (4.0 / 3.0) * target_edge_length;
    for (const auto edge : surface.edges()) {
        if (!surface.is_border(edge)) {
            continue;
        }
        const auto halfedge = surface.halfedge(edge);
        const auto source = surface.source(halfedge);
        const auto target = surface.target(halfedge);
        const auto difference = surface.point(target) - surface.point(source);
        if (std::hypot(difference.x(), difference.y(), difference.z()) > maximum_length) {
            throw std::invalid_argument("protected boundary edge exceeds 4/3 * target_edge_length");
        }
        fixed[source] = fixed[target] = true;
    }

    CGAL::Polygon_mesh_processing::isotropic_remeshing(
        surface.faces(), target_edge_length, surface,
        CGAL::parameters::number_of_iterations(static_cast<unsigned int>(num_iter))
            .protect_constraints(true)
            .vertex_is_constrained_map(fixed)
            // Keep fixed points out of tangential relaxation as well.
            .allow_move_functor(
                [fixed](Vertex vertex, const Surface::Point&, const Surface::Point&) { return !get(fixed, vertex); }));
    return detail::from_cgal_surface_mesh(surface);
}
} // namespace litepgo::mesh
