#include "litepgo/mesh/mesh_topology.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/mesh_helpers.hpp"
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
namespace {

template <int K> void bind_mesh_topology(nb::module_& m, const char* name) {
    using Topology = MeshTopology<K>;
    nb::class_<Topology>(m, name)
        .def(
            "__init__",
            [](Topology* self, int num_vertices, const Cells<K>& cells) {
                new (self) Topology(make_topology<K>(num_vertices, cells));
            },
            "num_vertices"_a, "cells"_a, "Owns cells (one row per cell) and preserves their order.")
        .def_prop_ro("vertices_per_cell", [](const Topology&) { return K; })
        .def_prop_ro("num_vertices", &Topology::num_vertices)
        .def_prop_ro("num_cells", &Topology::num_cells)
        .def_prop_ro("cells", &cells_of<K>, "Connectivity, one row per cell (a copy).")
        .def(
            "cell_vertices",
            [](const Topology& topology, int cell) {
                const auto& vertices = topology.cell_vertices(cell);
                return Eigen::Matrix<int, K, 1>(Eigen::Map<const Eigen::Matrix<int, K, 1>>(vertices.data()));
            },
            "cell"_a);
}
} // namespace

void bind_mesh_mesh_topology(nb::module_& m) {

    bind_mesh_topology<3>(m, "TriangleTopology");
    bind_mesh_topology<4>(m, "TetTopology");
    bind_mesh_topology<8>(m, "HexTopology");
    // The C++ alias: quads and tets share MeshTopology<4>.
    m.attr("QuadTopology") = m.attr("TetTopology");
}
} // namespace litepgo::mesh::binding
