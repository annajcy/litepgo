#pragma once

#include "litepgo/mesh/mesh_topology.hpp"

#include <array>
#include <vector>

namespace litepgo {

struct TriangleMidEdgeStencils {
    // Per face: its three vertices in connectivity order, then unique opposite
    // vertices across edges 0,1,2. Edge k is opposite local face vertex k.
    std::vector<std::vector<int>> vertices;
    // opposite[f][k] is a LOCAL row in vertices[f], or -1 at a boundary edge.
    std::vector<std::array<int, 3>> opposite;
};

// Requires consistently oriented, edge-manifold connectivity. Duplicate faces,
// equally directed shared edges, and edges with more than two faces are rejected.
// Owns its result; boundary geometry and boundary conditions are left to the caller.
TriangleMidEdgeStencils make_triangle_mid_edge_stencils(const TriangleTopology& topology);

} // namespace litepgo
