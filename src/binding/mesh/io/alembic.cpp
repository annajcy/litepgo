#include "litepgo/mesh/io/alembic.hpp"
#include "bindings.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
void bind_alembic(nb::module_& m) {
#ifdef LITEPGO_HAS_ALEMBIC
    m.attr("has_alembic") = true;
    m.def("write_abc_from_objs", &write_abc_from_objs, "path"_a, "frame_paths"_a, "fps"_a,
          nb::call_guard<nb::gil_scoped_release>());
#else
    m.attr("has_alembic") = false;
    m.def(
        "write_abc_from_objs",
        [](const std::filesystem::path&, const std::vector<std::filesystem::path>&, double) {
            throw std::runtime_error("Alembic export requires LITEPGO_WITH_ALEMBIC=ON");
        },
        "path"_a, "frame_paths"_a, "fps"_a);
#endif
}
} // namespace litepgo::mesh::binding
