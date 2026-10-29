#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/mesh/mesh_topology.hpp"

#include <variant>

namespace litepgo::mesh {

// Coordinates use vertex-major xyz order, matching VertexDofMap(topology, 3).
template <int K> struct MeshData {
    Vec positions;
    MeshTopology<K> topology;

    void validate() const {
        if (positions.size() != Eigen::Index(topology.num_vertices()) * 3 || !positions.allFinite())
            throw std::invalid_argument("MeshData requires three finite coordinates per vertex");
    }
};

using TriangleMesh = MeshData<3>;
using TetMesh = MeshData<4>;
using HexMesh = MeshData<8>;
using VolumeMesh = std::variant<TetMesh, HexMesh>;

} // namespace litepgo::mesh
