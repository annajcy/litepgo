#include "litepgo/fem/local/local_energy.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
template <int Dim> void bind_local_energy(nb::module_& m, const char* name) {
    using Local = LocalEnergyT<Dim>;
    using Coordinate = typename Local::Coordinate;
    nb::class_<Local>(m, name, "Pointwise density with derivatives with respect to one active coefficient block.")
        .def_prop_ro("num_dofs", &Local::num_dofs)
        .def_prop_ro("has_gradient", &Local::has_gradient)
        .def_prop_ro("has_hessian", &Local::has_hessian)
        .def("eval_value", &Local::eval_value, "xi"_a)
        .def("eval_gradient", &Local::eval_gradient, "xi"_a)
        .def("eval_hessian", &Local::eval_hessian, "xi"_a)
        .def(
            "accumulate",
            [](const Local& local, const Coordinate& xi, double weight, bool gradient, bool hessian) {
                Eigen::VectorXd g = Eigen::VectorXd::Zero(gradient ? local.num_dofs() : 0);
                Eigen::MatrixXd H =
                    Eigen::MatrixXd::Zero(hessian ? local.num_dofs() : 0, hessian ? local.num_dofs() : 0);
                const double value = local.accumulate(xi, weight, gradient ? &g : nullptr, hessian ? &H : nullptr);
                return value_and_derivatives(value, gradient, g, hessian, H);
            },
            "xi"_a, "weight"_a, "gradient"_a = false, "hessian"_a = false,
            "Weighted (value, gradient or None, Hessian or None) at xi; C++ adds these into caller buffers.");
}
} // namespace

void bind_fem_local_local_energy(nb::module_& m) {
    bind_local_energy<2>(m, "LocalEnergy2D");
    bind_local_energy<3>(m, "LocalEnergy3D");
}
} // namespace litepgo::fem::binding
