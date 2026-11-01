#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/bindings/bindings.hpp"
#include <nanobind/stl/filesystem.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
void bind_obj(nb::module_& m) {
    m.def("read_obj", &read_obj, "path"_a);
    m.def("write_obj", &write_obj, "path"_a, "mesh"_a);
}
} // namespace litepgo::mesh::binding
