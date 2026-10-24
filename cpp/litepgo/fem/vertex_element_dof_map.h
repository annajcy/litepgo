#pragma once

#include "litepgo/core/types.h"
#include "litepgo/fem/vertex_dof_layout.h"
#include "litepgo/mesh/mesh_topology.h"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace litepgo {

// Extracts each cell's vertex blocks in connectivity order, then slot order.
// Indices address one field vector, without offsets into a multi-field solver vector.
// The topology must outlive the map; the layout is copied.
template <int VerticesPerCell, int DofsPerVertex>
    requires(VerticesPerCell > 0 && DofsPerVertex > 0 &&
             VerticesPerCell <= std::numeric_limits<int>::max() / DofsPerVertex)
class VertexElementDofMap final {
  public:
    using Topology = MeshTopology<VerticesPerCell>;
    static constexpr int num_element_dofs = VerticesPerCell * DofsPerVertex;
    static constexpr int num_local_coefficients = num_element_dofs;

    using Indices = std::array<int, num_element_dofs>;
    using LocalCoefficients = Eigen::Matrix<double, num_local_coefficients, 1>;
    using CoefficientMatrix = Eigen::Matrix<double, num_local_coefficients, num_element_dofs>;
    using CoefficientJacobian = decltype(CoefficientMatrix::Identity());

    VertexElementDofMap(const MeshTopology<VerticesPerCell>& topology, VertexDofLayout<DofsPerVertex> layout)
        : topology_(topology), layout_(layout) {
        if (topology_.num_vertices() != layout_.num_vertices()) {
            throw std::invalid_argument("VertexElementDofMap requires matching topology and layout vertex counts");
        }
    }

    VertexElementDofMap(const MeshTopology<VerticesPerCell>&&, VertexDofLayout<DofsPerVertex>) = delete;

    int num_field_dofs() const noexcept {
        return layout_.num_dofs();
    }

    // Invalid element indices throw std::out_of_range.
    Indices dof_indices(int element) const {
        const auto& vertices = topology_.cell_vertices(element);
        Indices result;
        for (int i = 0; i < VerticesPerCell; ++i) {
            const int first = layout_.dof(vertices[static_cast<std::size_t>(i)], 0);
            for (int slot = 0; slot < DofsPerVertex; ++slot) {
                result[static_cast<std::size_t>(i * DofsPerVertex + slot)] = first + slot;
            }
        }
        return result;
    }

    // Copies coefficients without interpreting their values. A field vector of
    // another size throws std::invalid_argument; element bounds are checked too.
    LocalCoefficients gather(int element, const ConstVecRef& field_values) const {
        if (field_values.size() != num_field_dofs()) {
            throw std::invalid_argument("VertexElementDofMap::gather: field vector has the wrong size");
        }
        const auto indices = dof_indices(element);
        LocalCoefficients result;
        for (int i = 0; i < num_local_coefficients; ++i) {
            result[i] = field_values[indices[static_cast<std::size_t>(i)]];
        }
        return result;
    }

    // dc_e / d(field_values[dof_indices(element)]): an identity expression,
    // without allocating or storing a dense matrix.
    CoefficientJacobian coefficient_jacobian(int element) const {
        topology_.cell_vertices(element);
        return CoefficientMatrix::Identity();
    }

  private:
    const Topology& topology_;
    VertexDofLayout<DofsPerVertex> layout_;
};

} // namespace litepgo
