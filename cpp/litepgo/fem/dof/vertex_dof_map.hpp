#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/fem/dof/vertex_dof_data.hpp"
#include "litepgo/mesh/mesh_topology.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace litepgo {

// Extracts each cell's vertex blocks in connectivity order, then slot order.
// Indices address the supplied data, without offsets into a multi-field solver vector.
// Owns a connectivity snapshot; later topology changes do not affect this map.
class VertexDofMap final {
  public:
    template <int VerticesPerCell>
    explicit VertexDofMap(const MeshTopology<VerticesPerCell>& topology)
        : num_vertices_(topology.num_vertices()), cells_(topology.num_cells(), VerticesPerCell) {
        for (int e = 0; e < topology.num_cells(); ++e) {
            const auto& vertices = topology.cell_vertices(e);
            for (int i = 0; i < cells_.cols(); ++i) {
                cells_(e, i) = vertices[static_cast<std::size_t>(i)];
            }
        }
    }

    // A mismatched vertex count throws std::invalid_argument; an invalid
    // element index throws std::out_of_range. Checks are repeated for each data object.
    template <int DofsPerVertex>
    std::vector<int> dof_indices(int element, const VertexDofData<DofsPerVertex>& data) const {
        check(element, data);
        std::vector<int> result(static_cast<std::size_t>(cells_.cols()) * DofsPerVertex);
        for (int i = 0; i < cells_.cols(); ++i) {
            const int first = cells_(element, i) * DofsPerVertex;
            for (int slot = 0; slot < DofsPerVertex; ++slot) {
                result[static_cast<std::size_t>(i * DofsPerVertex + slot)] = first + slot;
            }
        }
        return result;
    }

    // Returns an independent copy: rows follow cell connectivity, columns are vertex slots.
    template <int DofsPerVertex>
    Eigen::Matrix<double, Eigen::Dynamic, DofsPerVertex> gather(int element,
                                                                const VertexDofData<DofsPerVertex>& data) const {
        const auto indices = dof_indices(element, data);
        const auto values = data.values();
        Eigen::Matrix<double, Eigen::Dynamic, DofsPerVertex> result(cells_.cols(), DofsPerVertex);
        for (int i = 0; i < cells_.cols(); ++i) {
            for (int slot = 0; slot < DofsPerVertex; ++slot) {
                result(i, slot) = values[indices[static_cast<std::size_t>(i * DofsPerVertex + slot)]];
            }
        }
        return result;
    }

  private:
    template <int DofsPerVertex> void check(int element, const VertexDofData<DofsPerVertex>& data) const {
        if (num_vertices_ != data.num_vertices()) {
            throw std::invalid_argument("VertexDofMap requires matching topology and data vertex counts");
        }
        if (element < 0 || element >= cells_.rows()) {
            throw std::out_of_range("VertexDofMap element index out of range");
        }
    }

    int num_vertices_;
    Eigen::MatrixXi cells_;
};

} // namespace litepgo
