#pragma once

#include "litepgo/core/types.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace litepgo {

// Owns one field's vertex-major DOF values. Counts come from the vector size
// so they stay consistent across copies and moves.
template <int DofsPerVertex>
    requires(DofsPerVertex > 0)
class VertexDofData final {
  public:
    // Creates zero data. The vertex count must be non-negative and the total
    // DOF count fit in int; otherwise throws std::invalid_argument.
    explicit VertexDofData(int num_vertices) : values_(Vec::Zero(checked_num_dofs(num_vertices))) {}

    // Copies data, inferring the vertex count. The size must be a multiple of
    // DofsPerVertex and fit in int; otherwise throws std::invalid_argument.
    explicit VertexDofData(const ConstVecRef& values) : values_(checked_size(values.size())) {
        values_ = values;
    }

    int num_vertices() const noexcept {
        return num_dofs() / DofsPerVertex;
    }
    static constexpr int dofs_per_vertex() noexcept {
        return DofsPerVertex;
    }
    int num_dofs() const noexcept {
        return static_cast<int>(values_.size());
    }

    // vertex * DofsPerVertex + slot. Throws std::out_of_range if either
    // index is invalid.
    int dof(int vertex, int slot) const {
        if (vertex < 0 || vertex >= num_vertices()) {
            throw std::out_of_range("VertexDofData::dof: vertex " + std::to_string(vertex) + " is out of range");
        }
        if (slot < 0 || slot >= DofsPerVertex) {
            throw std::out_of_range("VertexDofData::dof: slot " + std::to_string(slot) + " is out of range");
        }
        return vertex * DofsPerVertex + slot;
    }

    // Views permit editing values without resizing the owned vector. They are
    // valid until this object is assigned, moved from, or destroyed.
    ConstVecRef values() const {
        return values_;
    }
    VecRef values() {
        return values_;
    }

    double operator()(int vertex, int slot) const {
        return values_[dof(vertex, slot)];
    }
    double& operator()(int vertex, int slot) {
        return values_[dof(vertex, slot)];
    }

  private:
    static int checked_num_dofs(int num_vertices) {
        if (num_vertices < 0 || num_vertices > std::numeric_limits<int>::max() / DofsPerVertex) {
            throw std::invalid_argument(
                "VertexDofData requires a non-negative vertex count and a DOF count fitting in int");
        }
        return num_vertices * DofsPerVertex;
    }

    static Eigen::Index checked_size(Eigen::Index size) {
        if (size > std::numeric_limits<int>::max() || size % DofsPerVertex != 0) {
            throw std::invalid_argument(
                "VertexDofData requires a vector size divisible by its vertex width and fitting in int");
        }
        return size;
    }

    Vec values_;
};

} // namespace litepgo
