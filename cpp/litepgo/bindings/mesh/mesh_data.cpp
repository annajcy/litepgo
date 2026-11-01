#include "litepgo/mesh/mesh_data.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/mesh_helpers.hpp"
#include <nanobind/eigen/dense.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
namespace {

using Positions = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

template <int K> void bind_mesh_data(nb::module_& m, const char* name) {
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

void bind_mesh_mesh_data(nb::module_& m) {
    bind_mesh_data<3>(m, "TriangleMesh");
    bind_mesh_data<4>(m, "TetMesh");
    bind_mesh_data<8>(m, "HexMesh");
}
} // namespace litepgo::mesh::binding
