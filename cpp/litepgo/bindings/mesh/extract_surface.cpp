#include "litepgo/mesh/extract_surface.hpp"
#include "litepgo/bindings/bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
void bind_extract_surface(nb::module_& m) {
    m.def(
        "extract_surface", [](const VolumeMesh& mesh) { return extract_surface(mesh); }, "mesh"_a,
        "Outward-wound boundary triangles of a tet or hex mesh, keeping only boundary vertices.");

    nb::class_<ExtractedSurface>(m, "ExtractedSurface")
        .def_ro("mesh", &ExtractedSurface::mesh)
        .def_prop_ro("vertices", [](const ExtractedSurface& surface) {
            return Eigen::VectorXi(Eigen::Map<const Eigen::VectorXi>(surface.vertices.data(), surface.vertices.size()));
        });
    m.def(
        "extract_surface_with_vertices", [](const VolumeMesh& mesh) { return extract_surface_with_vertices(mesh); },
        "mesh"_a);
}
} // namespace litepgo::mesh::binding
