#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/sparse_matrix.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/surface_kinematics.hpp"
#include "litepgo/dof/vertex_displacement_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
namespace {
Eigen::VectorXi to_array(const std::vector<int>& indices) {
    return Eigen::Map<const Eigen::VectorXi>(indices.data(), static_cast<Eigen::Index>(indices.size()));
}

// Members shared by every DOF map.
template <class Map> nb::class_<Map> bind_map(nb::module_& m, const char* name) {
    return nb::class_<Map>(m, name)
        .def_prop_ro("num_elements", &Map::num_elements)
        .def_prop_ro("num_dofs", &Map::num_dofs)
        .def(
            "dof_indices", [](const Map& map, int element) { return to_array(map.dof_indices(element)); }, "element"_a,
            "Global DOFs of one element in local order (a copy).")
        .def("gather", &Map::gather, "element"_a, "x"_a, "The element's entries of a global vector.");
}

// The C++ constructors take any MeshTopology<K>.
template <class Map, class Cls> void def_topology_init(Cls& cls, const char* width) {
    cls.def(nb::init<const mesh::MeshTopology<3>&, int>(), "topology"_a, nb::arg(width))
        .def(nb::init<const mesh::MeshTopology<4>&, int>(), "topology"_a, nb::arg(width))
        .def(nb::init<const mesh::MeshTopology<8>&, int>(), "topology"_a, nb::arg(width));
}
} // namespace

void bind(nb::module_& m) {
    m.doc() = "Degree-of-freedom maps and reductions (cpp/litepgo/dof)";

    nb::class_<DofReduction>(m, "DofReduction")
        .def(nb::init<int, std::vector<int>>(), "num_dofs"_a, "fixed_dofs"_a)
        .def_prop_ro("num_dofs", &DofReduction::num_dofs)
        .def_prop_ro("num_free", &DofReduction::num_free)
        .def_prop_ro(
            "fixed_dofs",
            [](const DofReduction& reduction) {
                const auto dofs = reduction.fixed_dofs();
                return Eigen::VectorXi(Eigen::Map<const Eigen::VectorXi>(dofs.data(), std::ssize(dofs)));
            },
            "Fixed DOFs in increasing order (a copy).")
        .def_prop_ro(
            "free_dofs",
            [](const DofReduction& reduction) {
                const auto dofs = reduction.free_dofs();
                return Eigen::VectorXi(Eigen::Map<const Eigen::VectorXi>(dofs.data(), std::ssize(dofs)));
            },
            "Free DOFs in increasing order (a copy).")
        .def("free_index", &DofReduction::free_index, "dof"_a)
        .def("reduce", nb::overload_cast<const ConstVecRef&>(&DofReduction::reduce, nb::const_), "full"_a,
             "Free entries of a vector.")
        .def(
            "reduce",
            [](const DofReduction& reduction, const linalg::binding::SparseArgument& A) {
                return reduction.reduce(A.matrix);
            },
            "A"_a, "Principal submatrix of a scipy.sparse matrix on the free DOFs.")
        .def("expand", &DofReduction::expand, "reduced"_a, "full"_a.noconvert(),
             "Write reduced into the free entries of full, a contiguous writable float64 array.");

    m.def("build_vertex_displacement_map", &build_vertex_displacement_map, "vertices"_a, "num_dofs"_a,
          "dofs_per_vertex"_a = 3);

    using Positions = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    nb::class_<SurfaceKinematics>(m, "SurfaceKinematics")
        .def(
            "__init__",
            [](SurfaceKinematics* self, const Positions& rest_positions, const linalg::binding::SparseArgument& W) {
                new (self) SurfaceKinematics(rest_positions.reshaped<Eigen::RowMajor>(), W.matrix);
            },
            "rest_positions"_a, "displacement_map"_a,
            "p(u) = rest_positions + W u, from (n, 3) rest positions and a sparse (3n, num_dofs) W.")
        .def_prop_ro("num_vertices", &SurfaceKinematics::num_vertices)
        .def_prop_ro("num_dofs", &SurfaceKinematics::num_dofs)
        .def_prop_ro(
            "rest_positions",
            [](const SurfaceKinematics& map) -> Positions {
                return map.rest_positions().reshaped<Eigen::RowMajor>(map.num_vertices(), 3);
            },
            "One row per surface vertex (a copy).")
        .def_prop_ro(
            "displacement_map", [](const SurfaceKinematics& map) { return map.displacement_map(); },
            "W as a scipy.sparse.csc_matrix (a copy).")
        .def("displacements", &SurfaceKinematics::displacements, "u"_a, "W u, flat.")
        .def("positions", &SurfaceKinematics::positions, "u"_a, "rest positions + W u, flat (vertex-major xyz).");

    auto vertex =
        bind_map<VertexDofMap>(m, "VertexDofMap").def_prop_ro("dofs_per_vertex", &VertexDofMap::dofs_per_vertex);
    def_topology_init<VertexDofMap>(vertex, "dofs_per_vertex");

    auto element =
        bind_map<ElementDofMap>(m, "ElementDofMap").def_prop_ro("dofs_per_element", &ElementDofMap::dofs_per_element);
    def_topology_init<ElementDofMap>(element, "dofs_per_element");

    auto constant = bind_map<ConstantDofMap>(m, "ConstantDofMap");
    def_topology_init<ConstantDofMap>(constant, "num_dofs");

    bind_map<StencilDofMap>(m, "StencilDofMap")
        .def(nb::init<int, std::vector<std::vector<int>>, int>(), "num_vertices"_a, "stencils"_a, "dofs_per_vertex"_a)
        .def_prop_ro("dofs_per_vertex", &StencilDofMap::dofs_per_vertex)
        .def(
            "vertex_indices",
            [](const StencilDofMap& map, int element) { return to_array(map.vertex_indices(element)); }, "element"_a,
            "The element's stencil vertices (a copy).");
}

} // namespace litepgo::dof::binding
