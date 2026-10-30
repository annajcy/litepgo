#include "litepgo/dof/surface_dof_map.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace litepgo::dof {

namespace {

SpMat selection_map(Eigen::Index rows, const std::vector<int>& vertices, int num_dofs, int dofs_per_vertex) {
    if (static_cast<Eigen::Index>(vertices.size()) * 3 != rows) {
        throw std::invalid_argument("SurfaceDofMap requires one simulation vertex per surface vertex");
    }
    if (num_dofs < 0 || dofs_per_vertex < 3) {
        throw std::invalid_argument("SurfaceDofMap requires a non-negative DOF count and at least three DOFs per "
                                    "vertex");
    }
    std::vector<Triplet> entries;
    entries.reserve(3 * vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const int vertex = vertices[i];
        if (vertex == -1) {
            continue;
        }
        if (vertex < 0 || static_cast<long long>(dofs_per_vertex) * vertex + 3 > num_dofs) {
            throw std::invalid_argument("SurfaceDofMap: simulation vertex " + std::to_string(vertex) +
                                        " has no position DOFs among num_dofs");
        }
        for (int c = 0; c < 3; ++c) {
            entries.emplace_back(3 * static_cast<int>(i) + c, dofs_per_vertex * vertex + c, 1.0);
        }
    }
    SpMat map(rows, num_dofs);
    map.setFromTriplets(entries.begin(), entries.end());
    return map;
}

} // namespace

SurfaceDofMap::SurfaceDofMap(Vec rest_positions, SpMat displacement_map)
    : rest_positions_(std::move(rest_positions)), displacement_map_(std::move(displacement_map)) {
    if (rest_positions_.size() % 3 != 0 || rest_positions_.size() / 3 > std::numeric_limits<int>::max() / 3 ||
        !rest_positions_.allFinite()) {
        throw std::invalid_argument("SurfaceDofMap requires three finite rest coordinates per surface vertex");
    }
    if (displacement_map_.rows() != rest_positions_.size()) {
        throw std::invalid_argument("SurfaceDofMap: the displacement map must have 3 * num_vertices rows");
    }
    displacement_map_.makeCompressed();
    if (!Eigen::Map<const Vec>(displacement_map_.valuePtr(), displacement_map_.nonZeros()).allFinite()) {
        throw std::invalid_argument("SurfaceDofMap requires a finite displacement map");
    }
}

// rest_positions is copied, not moved: the order of the delegated arguments is unspecified.
SurfaceDofMap::SurfaceDofMap(Vec rest_positions, const std::vector<int>& vertices, int num_dofs, int dofs_per_vertex)
    : SurfaceDofMap(rest_positions, selection_map(rest_positions.size(), vertices, num_dofs, dofs_per_vertex)) {}

Vec SurfaceDofMap::displacements(const ConstVecRef& u) const {
    if (u.size() != num_dofs()) {
        throw std::invalid_argument("SurfaceDofMap: u must have num_dofs entries");
    }
    return displacement_map_ * u;
}

Vec SurfaceDofMap::positions(const ConstVecRef& u) const {
    return rest_positions_ + displacements(u);
}

} // namespace litepgo::dof
