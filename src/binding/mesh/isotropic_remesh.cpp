#include "litepgo/mesh/isotropic_remesh.hpp"
#include "bindings.hpp"

#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
void bind_isotropic_remesh(nb::module_& m) {
    m.def(
        "isotropic_remesh",
#ifdef LITEPGO_HAS_CGAL
        [](const TriangleMesh& mesh, double target_edge_length, int num_iter,
           const std::vector<int>& constrained_vertices) {
            return isotropic_remesh(mesh, target_edge_length, num_iter, constrained_vertices);
        },
#else
        [](const TriangleMesh&, double, int, const std::vector<int>&) -> TriangleMesh {
            throw std::runtime_error("Isotropic remeshing requires LITEPGO_WITH_CGAL=ON");
        },
#endif
        "mesh"_a, nb::kw_only(), "target_edge_length"_a, "num_iter"_a, "constrained_vertices"_a = std::vector<int>{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Remesh while preserving boundary edges and fixed vertex positions. Output indices may change.");
}
} // namespace litepgo::mesh::binding
