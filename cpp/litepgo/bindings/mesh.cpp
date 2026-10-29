#include "litepgo/bindings/bindings.hpp"
#include "litepgo/mesh/extract_surface.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#ifdef LITEPGO_HAS_GMSH
#include "litepgo/mesh/io/msh.hpp"
#endif
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
namespace {
template <int K> using Cells = Eigen::Matrix<int, Eigen::Dynamic, K, Eigen::RowMajor>;
using Positions = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

template <int K> MeshTopology<K> make_topology(int num_vertices, const Cells<K>& cells) {
    std::vector<typename MeshTopology<K>::Cell> connectivity(static_cast<std::size_t>(cells.rows()));
    for (Eigen::Index e = 0; e < cells.rows(); ++e)
        for (int j = 0; j < K; ++j)
            connectivity[static_cast<std::size_t>(e)][static_cast<std::size_t>(j)] = cells(e, j);
    return MeshTopology<K>(num_vertices, std::move(connectivity));
}

template <int K> Cells<K> cells_of(const MeshTopology<K>& topology) {
    Cells<K> cells(topology.num_cells(), K);
    for (int e = 0; e < topology.num_cells(); ++e)
        for (int j = 0; j < K; ++j)
            cells(e, j) = topology.cell_vertices(e)[static_cast<std::size_t>(j)];
    return cells;
}

template <int K> void bind_topology(nb::module_& m, const char* name) {
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

template <int K> void bind_mesh(nb::module_& m, const char* name) {
    using Mesh = MeshData<K>;
    nb::class_<Mesh>(m, name)
        .def(
            "__init__",
            [](Mesh* self, const Positions& positions, const Cells<K>& cells) {
                if (positions.rows() > std::numeric_limits<int>::max())
                    throw std::invalid_argument("Too many mesh vertices");
                Mesh mesh{positions.reshaped<Eigen::RowMajor>(),
                          make_topology<K>(static_cast<int>(positions.rows()), cells)};
                mesh.validate();
                new (self) Mesh(std::move(mesh));
            },
            "positions"_a, "cells"_a)
        .def(
            "__init__",
            [](Mesh* self, const Positions& positions, const MeshTopology<K>& topology) {
                Mesh mesh{positions.reshaped<Eigen::RowMajor>(), topology};
                mesh.validate();
                new (self) Mesh(std::move(mesh));
            },
            "positions"_a, "topology"_a)
        .def_prop_ro("positions",
                     [](const Mesh& mesh) -> Positions {
                         return Eigen::Map<const Positions>(mesh.positions.data(), mesh.topology.num_vertices(), 3);
                     })
        .def_prop_ro("cells", [](const Mesh& mesh) { return cells_of<K>(mesh.topology); })
        .def_prop_ro("topology", [](const Mesh& mesh) { return mesh.topology; }, "The connectivity (a copy).");
}
} // namespace

void bind(nb::module_& m) {
    m.doc() = "Mesh connectivity, geometry, and file I/O (cpp/litepgo/mesh)";

    bind_topology<3>(m, "TriangleTopology");
    bind_topology<4>(m, "TetTopology");
    bind_topology<8>(m, "HexTopology");
    // The C++ alias: quads and tets share MeshTopology<4>.
    m.attr("QuadTopology") = m.attr("TetTopology");
    bind_mesh<3>(m, "TriangleMesh");
    bind_mesh<4>(m, "TetMesh");
    bind_mesh<8>(m, "HexMesh");

    m.def(
        "extract_surface", [](const VolumeMesh& mesh) { return extract_surface(mesh); }, "mesh"_a,
        "Outward-wound boundary triangles of a tet or hex mesh, keeping only boundary vertices.");

    nb::class_<TriangleMidEdgeStencils>(m, "TriangleMidEdgeStencils")
        .def_ro("vertices", &TriangleMidEdgeStencils::vertices,
                "Per face: its three vertices, then the unique opposite vertices across edges 0, 1, 2.")
        .def_prop_ro(
            "opposite",
            [](const TriangleMidEdgeStencils& stencils) {
                Cells<3> opposite(static_cast<Eigen::Index>(stencils.opposite.size()), 3);
                for (std::size_t f = 0; f < stencils.opposite.size(); ++f)
                    for (int k = 0; k < 3; ++k)
                        opposite(static_cast<Eigen::Index>(f), k) = stencils.opposite[f][static_cast<std::size_t>(k)];
                return opposite;
            },
            "Per face: the local row across each edge, or -1 at a boundary (a copy).");
    m.def("make_triangle_mid_edge_stencils", &make_triangle_mid_edge_stencils, "topology"_a);

    m.def("read_obj", &read_obj, "path"_a);
    m.def("write_obj", &write_obj, "path"_a, "mesh"_a);
    m.def("read_veg", &read_veg, "path"_a);
    m.def("write_veg", &write_veg, "path"_a, "mesh"_a);
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
