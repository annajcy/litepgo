#pragma once

#include "litepgo/mesh/mesh_data.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

namespace litepgo::mesh::detail {
using CgalSurfaceMesh = CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3>;

// In the fresh surface, each vertex descriptor index equals its input vertex ID.
// Throws std::invalid_argument if CGAL cannot insert an input face.
CgalSurfaceMesh to_cgal_surface_mesh(const TriangleMesh& mesh);

// Converts live elements of a valid surface and preserves face winding. Output
// indices are compact. Throws std::invalid_argument for non-triangular faces.
TriangleMesh from_cgal_surface_mesh(const CgalSurfaceMesh& surface);
} // namespace litepgo::mesh::detail
