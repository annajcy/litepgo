#include "litepgo/fem/material/shell/shell_fp_force_local_response.hpp"
#include "bindings.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::fem::binding {
void bind_fem_material_shell_shell_fp_force_local_response(nb::module_& m) {
    using Local = ShellFpForceLocalResponse2D<KoiterStVK>;
    nb::class_<Local, LocalResponse2D>(m, "ShellFpForceLocalResponse2D[KoiterStVK]")
        .def(
            nb::init<const TriangleMidEdgeRestData&, const Eigen::MatrixXd&, double, const Field2D&, Eigen::VectorXd>(),
            "rest"_a, "u_local"_a, "thickness"_a, "Fp"_a, "theta"_a, nb::keep_alive<1, 5>(),
            "Fixed original geometry, displacement, thickness and theta; keeps the Fp field alive.");
    bindings::register_instance(m, "ShellFpForceLocalResponse2D", nb::make_tuple(nb::type<KoiterStVK>()),
                                nb::type<Local>());
}
} // namespace litepgo::fem::binding
