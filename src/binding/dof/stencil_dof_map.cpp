#include "litepgo/dof/stencil_dof_map.hpp"
#include "bindings.hpp"
#include "detail/dof_helpers.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
void bind_stencil_dof_map(nb::module_& m) {
    bind_map<StencilDofMap>(m, "StencilDofMap")
        .def(nb::init<int, std::vector<std::vector<int>>, int>(), "num_vertices"_a, "stencils"_a, "dofs_per_vertex"_a)
        .def_prop_ro("dofs_per_vertex", &StencilDofMap::dofs_per_vertex)
        .def(
            "vertex_indices",
            [](const StencilDofMap& map, int element) { return to_array(map.vertex_indices(element)); }, "element"_a,
            "The element's stencil vertices (a copy).");
}
} // namespace litepgo::dof::binding
