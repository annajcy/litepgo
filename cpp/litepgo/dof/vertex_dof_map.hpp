#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/mesh/mesh_topology.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace litepgo {

// Extracts each cell's vertex blocks in connectivity order, then slot order.
// Indices address the supplied data, without offsets into a multi-field solver vector.
// Owns a connectivity snapshot; later topology changes do not affect this map.
class VertexDofMap final {
  public:
    template <int VerticesPerCell>
    explicit VertexDofMap(const MeshTopology<VerticesPerCell>& topology, int dofs_per_vertex)
        : num_vertices_(topology.num_vertices()), dofs_per_vertex_(dofs_per_vertex),
          cells_(topology.num_cells(), VerticesPerCell) {
        if (dofs_per_vertex <= 0 || num_vertices_ > std::numeric_limits<int>::max() / dofs_per_vertex)
            throw std::invalid_argument("VertexDofMap requires positive width and total DOFs fitting in int");
        for (int e = 0; e < topology.num_cells(); ++e) {
            const auto& vertices = topology.cell_vertices(e);
            for (int i = 0; i < cells_.cols(); ++i) {
                cells_(e, i) = vertices[static_cast<std::size_t>(i)];
            }
        }
    }

    int num_elements() const noexcept {
        return static_cast<int>(cells_.rows());
    }

    int num_dofs() const noexcept {
        return num_vertices_ * dofs_per_vertex_;
    }
    int dofs_per_vertex() const noexcept {
        return dofs_per_vertex_;
    }

    std::vector<int> dof_indices(int element) const {
        check_element(element);
        std::vector<int> result(static_cast<std::size_t>(cells_.cols()) * dofs_per_vertex_);
        for (int i = 0; i < cells_.cols(); ++i)
            for (int c = 0; c < dofs_per_vertex_; ++c)
                result[static_cast<std::size_t>(i * dofs_per_vertex_ + c)] = cells_(element, i) * dofs_per_vertex_ + c;
        return result;
    }

    // Independent copy: rows follow connectivity, columns are vertex DOF slots.
    Eigen::MatrixXd gather(int element, const ConstVecRef& x) const {
        check_element(element);
        if (x.size() != num_dofs())
            throw std::invalid_argument("VertexDofMap: incorrect global vector size");
        Eigen::MatrixXd x_local(cells_.cols(), dofs_per_vertex_);
        for (int i = 0; i < cells_.cols(); ++i)
            x_local.row(i) = x.segment(cells_(element, i) * dofs_per_vertex_, dofs_per_vertex_).transpose();
        return x_local;
    }

  private:
    void check_element(int element) const {
        if (element < 0 || element >= num_elements())
            throw std::out_of_range("VertexDofMap: element out of range");
    }

    int num_vertices_;
    int dofs_per_vertex_;
    Eigen::MatrixXi cells_;
};

} // namespace litepgo
