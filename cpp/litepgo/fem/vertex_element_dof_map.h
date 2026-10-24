#pragma once

#include "litepgo/core/types.h"
#include "litepgo/fem/vertex_dof_data.h"
#include "litepgo/mesh/mesh_topology.h"

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
    using Topology = MeshTopology<VerticesPerCell>;

    template <int DofsPerVertex>
        requires(DofsPerVertex > 0 && VerticesPerCell <= std::numeric_limits<int>::max() / DofsPerVertex)
    static constexpr int num_element_dofs = VerticesPerCell * DofsPerVertex;

    template <int DofsPerVertex> using Indices = std::array<int, num_element_dofs<DofsPerVertex>>;
    template <int DofsPerVertex>
        requires(DofsPerVertex > 0 && VerticesPerCell <= std::numeric_limits<int>::max() / DofsPerVertex)
    using LocalCoefficients = Eigen::Matrix<double, VerticesPerCell, DofsPerVertex>;
    template <int DofsPerVertex>
    using CoefficientMatrix = Eigen::Matrix<double, num_element_dofs<DofsPerVertex>, num_element_dofs<DofsPerVertex>>;
    template <int DofsPerVertex> using CoefficientJacobian = decltype(CoefficientMatrix<DofsPerVertex>::Identity());

    explicit VertexElementDofMap(const MeshTopology<VerticesPerCell>& topology) : topology_(topology) {}

    VertexElementDofMap(const MeshTopology<VerticesPerCell>&&) = delete;

    // A mismatched vertex count throws std::invalid_argument; an invalid
    // element index throws std::out_of_range. Checks are repeated for each data object.
    template <int DofsPerVertex>
    Indices<DofsPerVertex> dof_indices(int element, const VertexDofData<DofsPerVertex>& data) const {
        check_vertex_count(data);
        const auto& vertices = topology_.cell_vertices(element);
        Indices<DofsPerVertex> result;
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
    LocalCoefficients<DofsPerVertex> gather(int element, const VertexDofData<DofsPerVertex>& data) const {
        const auto indices = dof_indices(element, data);
        const auto values = data.values();
        LocalCoefficients<DofsPerVertex> result;
        for (int i = 0; i < VerticesPerCell; ++i) {
            for (int slot = 0; slot < DofsPerVertex; ++slot) {
                result(i, slot) = values[indices[static_cast<std::size_t>(i * DofsPerVertex + slot)]];
            }
        }
        return result;
    }

    // d(flatten_rows(c_e)) / d(data.values()[dof_indices(element, data)]):
    // an identity expression, without allocating or storing a dense matrix.
    template <int DofsPerVertex>
    CoefficientJacobian<DofsPerVertex> coefficient_jacobian(int element,
                                                            const VertexDofData<DofsPerVertex>& data) const {
        check_vertex_count(data);
        topology_.cell_vertices(element);
        return CoefficientMatrix<DofsPerVertex>::Identity();
    }

  private:
    template <int DofsPerVertex> void check_vertex_count(const VertexDofData<DofsPerVertex>& data) const {
        if (topology_.num_vertices() != data.num_vertices()) {
            throw std::invalid_argument("VertexElementDofMap requires matching topology and data vertex counts");
        }
    }

    const Topology& topology_;
};

} // namespace litepgo
