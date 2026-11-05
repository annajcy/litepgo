#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/mesh/mesh_topology.hpp"
#include <array>
#include <vector>

namespace litepgo::contact {

/// Fixed primitive groups and symmetric, finite, nonnegative contact multipliers.
/// Energies own their configuration. Zero excludes a candidate from the active
/// set and CCD; the independent whole-surface intersection check remains strict.
struct ContactPairWeights {
    std::vector<int> vertex_groups; ///< One group per input vertex.
    std::vector<int> edge_groups;   ///< One group per row of contact_edges(topology).
    std::vector<int> face_groups;   ///< One group per input triangle.
    Eigen::MatrixXd weights;        ///< G x G; group IDs must be in [0, G).
};

/// Unique undirected edges (min vertex, max vertex), sorted lexicographically.
std::vector<std::array<int, 2>> contact_edges(const mesh::TriangleTopology& topology);

} // namespace litepgo::contact
