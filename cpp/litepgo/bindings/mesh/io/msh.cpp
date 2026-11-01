#include "litepgo/bindings/bindings.hpp"
#include "litepgo/mesh/mesh_data.hpp"
#include <filesystem>
#ifdef LITEPGO_HAS_GMSH
#include "litepgo/mesh/io/msh.hpp"
#endif
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/variant.h>

#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
void bind_msh(nb::module_& m) {
#ifdef LITEPGO_HAS_GMSH
    m.attr("has_gmsh") = true;
    m.def("read_msh", &read_msh, "path"_a);
    m.def("write_msh", &write_msh, "path"_a, "mesh"_a);
#else
    m.attr("has_gmsh") = false;
    m.def(
        "read_msh",
        [](const std::filesystem::path&) { throw std::runtime_error("MSH I/O requires LITEPGO_WITH_GMSH=ON"); },
        "path"_a);
    m.def(
        "write_msh",
        [](const std::filesystem::path&, const VolumeMesh&) {
            throw std::runtime_error("MSH I/O requires LITEPGO_WITH_GMSH=ON");
        },
        "path"_a, "mesh"_a);
#endif
}
} // namespace litepgo::mesh::binding
