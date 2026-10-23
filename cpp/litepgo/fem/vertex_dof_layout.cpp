#include "litepgo/fem/vertex_dof_layout.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace litepgo {

VertexDofLayout::VertexDofLayout(int num_vertices, int dofs_per_vertex)
    : num_vertices_(num_vertices), dofs_per_vertex_(dofs_per_vertex) {
    if (num_vertices_ < 0 || dofs_per_vertex_ <= 0) {
        throw std::invalid_argument("VertexDofLayout requires a non-negative vertex count and positive block size");
    }
    if (num_vertices_ > std::numeric_limits<int>::max() / dofs_per_vertex_) {
        throw std::invalid_argument("VertexDofLayout DOF count exceeds the int index range");
    }
}

int VertexDofLayout::dof(int vertex, int slot) const {
    if (vertex < 0 || vertex >= num_vertices_) {
        throw std::out_of_range("VertexDofLayout::dof: vertex " + std::to_string(vertex) + " is out of range");
    }
    if (slot < 0 || slot >= dofs_per_vertex_) {
        throw std::out_of_range("VertexDofLayout::dof: slot " + std::to_string(slot) + " is out of range");
    }
    return vertex * dofs_per_vertex_ + slot;
}

} // namespace litepgo
