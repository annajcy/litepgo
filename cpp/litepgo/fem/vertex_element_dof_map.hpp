#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/fem/vertex_dof_data.hpp"
#include "litepgo/mesh/mesh_topology.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace litepgo {

// Extracts each cell's vertex blocks in connectivity order, then slot order.
// Indices address the supplied data, without offsets into a multi-field solver vector.
// The topology must outlive the map.
template <int VerticesPerCell>
    requires(VerticesPerCell > 0)
class VertexElementDofMap final {
  public:
    template <int DofsPerVertex>
        requires(DofsPerVertex > 0 && VerticesPerCell <= std::numeric_limits<int>::max() / DofsPerVertex)
    static constexpr int num_element_dofs = VerticesPerCell * DofsPerVertex;

    explicit VertexElementDofMap(const MeshTopology<VerticesPerCell>& topology) : topology_(topology) {}

    VertexElementDofMap(const MeshTopology<VerticesPerCell>&&) = delete;

    // A mismatched vertex count throws std::invalid_argument; an invalid
    // element index throws std::out_of_range. Checks are repeated for each data object.
    template <int DofsPerVertex>
    std::array<int, num_element_dofs<DofsPerVertex>> dof_indices(int element,
                                                                 const VertexDofData<DofsPerVertex>& data) const {
        check_vertex_count(data);
        const auto& vertices = topology_.cell_vertices(element);
        std::array<int, num_element_dofs<DofsPerVertex>> result;
        for (int i = 0; i < VerticesPerCell; ++i) {
            const int first = data.dof(vertices[static_cast<std::size_t>(i)], 0);
            for (int slot = 0; slot < DofsPerVertex; ++slot) {
                result[static_cast<std::size_t>(i * DofsPerVertex + slot)] = first + slot;
            }
        }
        return result;
    }

    // Returns an independent copy: rows follow cell connectivity, columns are vertex slots.
    template <int DofsPerVertex>
        requires(DofsPerVertex > 0 && VerticesPerCell <= std::numeric_limits<int>::max() / DofsPerVertex)
    Eigen::Matrix<double, VerticesPerCell, DofsPerVertex> gather(int element,
                                                                 const VertexDofData<DofsPerVertex>& data) const {
        const auto indices = dof_indices(element, data);
        const auto values = data.values();
        Eigen::Matrix<double, VerticesPerCell, DofsPerVertex> result;
        for (int i = 0; i < VerticesPerCell; ++i) {
            for (int slot = 0; slot < DofsPerVertex; ++slot) {
                result(i, slot) = values[indices[static_cast<std::size_t>(i * DofsPerVertex + slot)]];
            }
        }
        return result;
    }

  private:
    template <int DofsPerVertex> void check_vertex_count(const VertexDofData<DofsPerVertex>& data) const {
        if (topology_.num_vertices() != data.num_vertices()) {
            throw std::invalid_argument("VertexElementDofMap requires matching topology and data vertex counts");
        }
    }

    const MeshTopology<VerticesPerCell>& topology_;
};

} // namespace litepgo
