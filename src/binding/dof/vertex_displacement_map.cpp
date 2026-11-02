#include "litepgo/dof/vertex_displacement_map.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
void bind_vertex_displacement_map(nb::module_& m) {
    m.def("build_vertex_displacement_map", &build_vertex_displacement_map, "vertices"_a, "num_dofs"_a,
          "dofs_per_vertex"_a = 3);
}
} // namespace litepgo::dof::binding
