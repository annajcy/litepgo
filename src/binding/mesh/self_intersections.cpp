#include "litepgo/mesh/self_intersections.hpp"
#include "bindings.hpp"

#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
void bind_self_intersections(nb::module_& m) {
#ifdef LITEPGO_HAS_CGAL
    m.attr("has_cgal") = true;
    m.def("has_self_intersections", &has_self_intersections, "mesh"_a, nb::call_guard<nb::gil_scoped_release>(),
          "Detect self-intersections or degenerate triangles. Shared indexed edges and vertices may touch.");
#else
    m.attr("has_cgal") = false;
    m.def(
        "has_self_intersections",
        [](const TriangleMesh&) -> bool {
            throw std::runtime_error("Self-intersection detection requires LITEPGO_WITH_CGAL=ON");
        },
        "mesh"_a);
#endif
}
} // namespace litepgo::mesh::binding
