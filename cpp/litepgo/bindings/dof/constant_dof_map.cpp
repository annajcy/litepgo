#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/dof_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
void bind_constant_dof_map(nb::module_& m) {
    auto constant = bind_map<ConstantDofMap>(m, "ConstantDofMap");
    def_topology_init(constant, "num_dofs");
}
} // namespace litepgo::dof::binding
