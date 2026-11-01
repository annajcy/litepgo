#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/bindings/bindings.hpp"
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
void bind_veg(nb::module_& m) {
    m.def("read_veg", &read_veg, "path"_a);
    m.def("write_veg", &write_veg, "path"_a, "mesh"_a);
}
} // namespace litepgo::mesh::binding
