#include "litepgo/dof/surface_kinematics.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
void bind_surface_kinematics(nb::module_& m) {
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
}
} // namespace litepgo::dof::binding
