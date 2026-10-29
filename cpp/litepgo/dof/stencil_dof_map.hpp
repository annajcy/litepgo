#pragma once

#include "litepgo/core/types.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litepgo::dof {

// Owns variable-length vertex stencils. Rows retain the supplied vertex order;
// scalar DOFs use vertex-major, then slot order, without multi-field offsets.
class StencilDofMap final {
  public:
    // Each stencil must contain distinct vertices in [0, num_vertices).
    // Empty stencils, empty meshes, and unused vertices are allowed.
    StencilDofMap(int num_vertices, std::vector<std::vector<int>> stencils, int dofs_per_vertex)
        : num_vertices_(num_vertices), dofs_per_vertex_(dofs_per_vertex), stencils_(std::move(stencils)) {
        const int max_count = std::numeric_limits<int>::max();
        if (num_vertices_ < 0 || dofs_per_vertex_ <= 0 || num_vertices_ > max_count / dofs_per_vertex_ ||
            stencils_.size() > static_cast<std::size_t>(max_count))
            throw std::invalid_argument("StencilDofMap: invalid vertex, DOF, or element count");
        for (const auto& vertices : stencils_) {
            if (vertices.size() > static_cast<std::size_t>(max_count / dofs_per_vertex_))
                throw std::invalid_argument("StencilDofMap: local DOF count must fit in int");
            auto sorted = vertices;
            std::sort(sorted.begin(), sorted.end());
            if (!sorted.empty() && (sorted.front() < 0 || sorted.back() >= num_vertices_))
                throw std::invalid_argument("StencilDofMap: vertex index out of range");
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
                throw std::invalid_argument("StencilDofMap: stencil vertices must be distinct");
        }
    }

    int num_elements() const noexcept {
        return static_cast<int>(stencils_.size());
    }
    int num_dofs() const noexcept {
        return num_vertices_ * dofs_per_vertex_;
    }
    int dofs_per_vertex() const noexcept {
        return dofs_per_vertex_;
    }

    const std::vector<int>& vertex_indices(int element) const {
        if (element < 0 || element >= num_elements())
            throw std::out_of_range("StencilDofMap: element out of range");
        return stencils_[static_cast<std::size_t>(element)];
    }

    std::vector<int> dof_indices(int element) const {
        const auto& vertices = vertex_indices(element);
        std::vector<int> result(vertices.size() * dofs_per_vertex_);
        for (std::size_t i = 0; i < vertices.size(); ++i)
            for (int c = 0; c < dofs_per_vertex_; ++c)
                result[i * dofs_per_vertex_ + c] = vertices[i] * dofs_per_vertex_ + c;
        return result;
    }

    // Independent copy with one row per stencil vertex and one column per slot.
    Eigen::MatrixXd gather(int element, const ConstVecRef& x) const {
        const auto& vertices = vertex_indices(element);
        if (x.size() != num_dofs())
            throw std::invalid_argument("StencilDofMap: incorrect global vector size");
        Eigen::MatrixXd local(static_cast<Eigen::Index>(vertices.size()), dofs_per_vertex_);
        for (std::size_t i = 0; i < vertices.size(); ++i)
            local.row(static_cast<Eigen::Index>(i)) =
                x.segment(vertices[i] * dofs_per_vertex_, dofs_per_vertex_).transpose();
        return local;
    }

  private:
    int num_vertices_;
    int dofs_per_vertex_;
    std::vector<std::vector<int>> stencils_;
};

} // namespace litepgo::dof
