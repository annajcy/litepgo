#pragma once

#include "litepgo/fem/vertex_dof_layout.h"

namespace litepgo {

// Three displacement components per vertex, ordered x, y, z as in libpgo.
class DisplacementDofLayout final : public VertexDofLayout {
  public:
    explicit DisplacementDofLayout(int num_vertices) : VertexDofLayout(num_vertices, 3) {}

    // component is 0 (x), 1 (y), or 2 (z).
    int displacement_dof(int vertex, int component) const {
        return dof(vertex, component);
    }
};

} // namespace litepgo
