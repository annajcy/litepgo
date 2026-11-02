#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "bindings.hpp"
#include <array>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_triangle_mid_edge_kinematics(nb::module_& m) {
    nb::class_<TriangleMidEdgeRestData>(m, "TriangleMidEdgeRestData")
        .def(nb::init<Eigen::MatrixXd, std::array<int, 3>>(), "X"_a, "opposite_rows"_a,
             "Reference stencil positions (central face first) and the local row across each edge, or -1.")
        .def_ro("X_local", &TriangleMidEdgeRestData::X_local)
        .def_ro("opposite", &TriangleMidEdgeRestData::opposite)
        .def_ro("a_bar", &TriangleMidEdgeRestData::a_bar)
        .def_ro("b_bar", &TriangleMidEdgeRestData::b_bar)
        .def_ro("sqrt_det_a_bar", &TriangleMidEdgeRestData::sqrt_det_a_bar);

    nb::class_<TriangleMidEdgeKinematics>(m, "TriangleMidEdgeKinematics")
        .def_ro("a", &TriangleMidEdgeKinematics::a, "First fundamental form.")
        .def_ro("b", &TriangleMidEdgeKinematics::b, "Second fundamental form, b_ab = n.dot(x_,ab).");
    m.def(
        "eval_triangle_mid_edge_kinematics",
        [](const Eigen::MatrixXd& x, const std::array<int, 3>& opposite, bool gradient, bool hessian) {
            Eigen::MatrixXd dab_dx;
            std::array<Eigen::MatrixXd, 6> d2ab_dx2;
            const auto kinematics = eval_triangle_mid_edge_kinematics(x, opposite, gradient ? &dab_dx : nullptr,
                                                                      hessian ? &d2ab_dx2 : nullptr);
            return nb::make_tuple(kinematics, gradient ? nb::cast(dab_dx) : nb::none(),
                                  hessian ? nb::cast(d2ab_dx2) : nb::none());
        },
        "x"_a, "opposite"_a, "gradient"_a = false, "hessian"_a = false,
        "(kinematics, dab_dx or None, d2ab_dx2 or None); rows/components use (a11,a12,a22,b11,b12,b22) and x "
        "flattens by row.");
    m.def("eval_triangle_mid_edge_feasible_step", &eval_triangle_mid_edge_feasible_step, "x"_a, "dx"_a, "opposite"_a);
}
} // namespace litepgo::fem::binding
