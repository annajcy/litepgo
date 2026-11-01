#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/dof_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
void bind_vertex_dof_map(nb::module_& m) {
    auto vertex =
        bind_map<VertexDofMap>(m, "VertexDofMap").def_prop_ro("dofs_per_vertex", &VertexDofMap::dofs_per_vertex);
    def_topology_init(vertex, "dofs_per_vertex");
}
} // namespace litepgo::dof::binding
