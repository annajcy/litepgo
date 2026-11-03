#pragma once

#include "litepgo/mesh/mesh_data.hpp"

namespace litepgo::mesh {
// Requires litepgo::mesh_cgal. Contact confined to a shared indexed vertex or
// edge is allowed. Degenerate triangles are reported as self-intersections.
bool has_self_intersections(const TriangleMesh& mesh);
} // namespace litepgo::mesh
