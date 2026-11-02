#include "litepgo/dof/element_dof_map.hpp"
#include "bindings.hpp"
#include "detail/dof_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
void bind_element_dof_map(nb::module_& m) {
    auto element =
        bind_map<ElementDofMap>(m, "ElementDofMap").def_prop_ro("dofs_per_element", &ElementDofMap::dofs_per_element);
    def_topology_init(element, "dofs_per_element");
}
} // namespace litepgo::dof::binding
