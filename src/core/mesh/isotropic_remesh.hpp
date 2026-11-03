#pragma once

#include "litepgo/mesh/mesh_data.hpp"

#include <span>

namespace litepgo::mesh {
// Requires litepgo::mesh_cgal. Keeps boundary edges and constrained vertex
// positions fixed. Boundary edges must be at most 4/3 * target_edge_length.
// Output vertex and face indices may change.
TriangleMesh isotropic_remesh(const TriangleMesh& mesh, double target_edge_length, int num_iter,
                              std::span<const int> constrained_vertices = {});
} // namespace litepgo::mesh
