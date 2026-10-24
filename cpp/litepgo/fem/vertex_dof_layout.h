#pragma once

#include <limits>
#include <stdexcept>
#include <string>

namespace litepgo {

// Consecutive scalar DOF blocks in one field vector. Stores counts only;
// the field and its element map give the slots their meaning.
template <int DofsPerVertex>
    requires(DofsPerVertex > 0)
class VertexDofLayout final {
  public:
    // The vertex count must be non-negative and the total DOF count fit in int.
    explicit VertexDofLayout(int num_vertices) : num_vertices_(num_vertices) {
        if (num_vertices_ < 0 || num_vertices_ > std::numeric_limits<int>::max() / DofsPerVertex) {
            throw std::invalid_argument(
                "VertexDofLayout requires a non-negative vertex count and a DOF count fitting in int");
        }
    }

    int num_vertices() const noexcept {
        return num_vertices_;
    }
    static constexpr int dofs_per_vertex() noexcept {
        return DofsPerVertex;
    }
    int num_dofs() const noexcept {
        return num_vertices_ * DofsPerVertex;
    }

    // vertex * DofsPerVertex + slot. Throws std::out_of_range if either
    // index is invalid.
    int dof(int vertex, int slot) const {
        if (vertex < 0 || vertex >= num_vertices_) {
            throw std::out_of_range("VertexDofLayout::dof: vertex " + std::to_string(vertex) + " is out of range");
        }
        if (slot < 0 || slot >= DofsPerVertex) {
            throw std::out_of_range("VertexDofLayout::dof: slot " + std::to_string(slot) + " is out of range");
        }
        return vertex * DofsPerVertex + slot;
    }

  private:
    int num_vertices_;
};

} // namespace litepgo
