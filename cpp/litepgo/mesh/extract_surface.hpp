#pragma once
#include "litepgo/mesh/mesh_data.hpp"

namespace litepgo::mesh {
// Boundary of a conforming volume mesh as triangles, for visualization: cell faces
// that belong to exactly one cell, with hex quads split along one diagonal. Faces
// keep cell order and wind outward for positively oriented cells (TetP1Basis and
// HexQ1Basis node order). Only boundary vertices are kept, in increasing original
// index. A face shared by more than two cells throws std::invalid_argument.
TriangleMesh extract_surface(const TetMesh& mesh);
TriangleMesh extract_surface(const HexMesh& mesh);
TriangleMesh extract_surface(const VolumeMesh& mesh);
} // namespace litepgo::mesh
