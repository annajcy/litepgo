#pragma once

namespace litepgo {

// Consecutive scalar DOF blocks, one per vertex. Subclasses give slots their
// field-specific meaning. Holds counts only, with no coordinates or DOF values.
class VertexDofLayout {
  public:
    virtual ~VertexDofLayout() = default;

    int num_vertices() const noexcept {
        return num_vertices_;
    }
    int dofs_per_vertex() const noexcept {
        return dofs_per_vertex_;
    }
    int num_dofs() const noexcept {
        return num_vertices_ * dofs_per_vertex_;
    }

    // vertex * dofs_per_vertex + slot. Throws std::out_of_range if either
    // index is invalid.
    int dof(int vertex, int slot) const;

  protected:
    // Vertex count must be non-negative, block size positive, and the total
    // DOF count representable as int; otherwise throws std::invalid_argument.
    VertexDofLayout(int num_vertices, int dofs_per_vertex);

  private:
    int num_vertices_;
    int dofs_per_vertex_;
};

} // namespace litepgo
