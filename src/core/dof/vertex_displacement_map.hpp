#pragma once
#include "litepgo/common/types.hpp"
#include <limits>
#include <stdexcept>
#include <vector>

namespace litepgo::dof {
// Vertex-major physical DOFs: select the first three (value) slots exactly.
inline SpMat build_vertex_displacement_map(const std::vector<int>& vertices, int num_dofs, int dofs_per_vertex = 3) {
    if (num_dofs < 0 || dofs_per_vertex < 3 || vertices.size() > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument("build_vertex_displacement_map: invalid dimensions");
    std::vector<Triplet> entries;
    entries.reserve(3 * vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const int v = vertices[i];
        if (v < 0 || static_cast<long long>(v) * dofs_per_vertex + 3 > num_dofs)
            throw std::out_of_range("build_vertex_displacement_map: vertex has no position DOFs");
        for (int c = 0; c < 3; ++c)
            entries.emplace_back(3 * static_cast<int>(i) + c, v * dofs_per_vertex + c, 1.0);
    }
    SpMat W(3 * static_cast<int>(vertices.size()), num_dofs);
    W.setFromTriplets(entries.begin(), entries.end());
    return W;
}
} // namespace litepgo::dof
