#include "litepgo/bindings/bindings.hpp"
#include "litepgo/io/obj.hpp"
#include "litepgo/io/veg.hpp"
#ifdef LITEPGO_HAS_GMSH
#include "litepgo/io/msh.hpp"
#endif
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::bindings {
namespace {
template <int K> void bind_mesh(nb::module_& m, const char* name) {
    using Mesh = MeshData<K>;
    using Positions = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Cells = Eigen::Matrix<int, Eigen::Dynamic, K, Eigen::RowMajor>;
    nb::class_<Mesh>(m, name)
        .def(
            "__init__",
            [](Mesh* self, const Positions& positions, const Cells& cells) {
                if (positions.rows() > std::numeric_limits<int>::max())
                    throw std::invalid_argument("Too many mesh vertices");
                std::vector<typename MeshTopology<K>::Cell> connectivity(cells.rows());
                for (Eigen::Index e = 0; e < cells.rows(); ++e)
                    for (int j = 0; j < K; ++j)
                        connectivity[e][j] = cells(e, j);
                Mesh mesh{positions.reshaped<Eigen::RowMajor>(),
                          MeshTopology<K>(static_cast<int>(positions.rows()), std::move(connectivity))};
                mesh.validate();
                new (self) Mesh(std::move(mesh));
            },
            "positions"_a, "cells"_a)
        .def_prop_ro("positions",
                     [](const Mesh& mesh) -> Positions {
                         return Eigen::Map<const Positions>(mesh.positions.data(), mesh.topology.num_vertices(), 3);
                     })
        .def_prop_ro("cells", [](const Mesh& mesh) {
            Cells cells(mesh.topology.num_cells(), K);
            for (int e = 0; e < cells.rows(); ++e)
                for (int j = 0; j < K; ++j)
                    cells(e, j) = mesh.topology.cell_vertices(e)[j];
            return cells;
        });
}
} // namespace
void bind_io(nb::module_& m) {
#ifdef LITEPGO_HAS_GMSH
    m.attr("has_gmsh") = true;
    m.def("read_msh", &io::read_msh, "path"_a);
    m.def("write_msh", &io::write_msh, "path"_a, "mesh"_a);
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
    bind_mesh<3>(m, "TriangleMesh");
    bind_mesh<4>(m, "TetMesh");
    bind_mesh<8>(m, "HexMesh");
    m.def("read_obj", &io::read_obj, "path"_a);
    m.def("write_obj", &io::write_obj, "path"_a, "mesh"_a);
    m.def("read_veg", &io::read_veg, "path"_a);
    m.def("write_veg", &io::write_veg, "path"_a, "mesh"_a);
}
} // namespace litepgo::bindings
