#include "litepgo/fem/local/local_response.hpp"
#include "bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
template <int Dim> void bind_local_response(nb::module_& m, const char* name) {
    using Local = LocalResponseT<Dim>;
    using Coordinate = typename Local::Coordinate;
    nb::class_<Local>(m, name, "Pointwise vector density and a rectangular coefficient Jacobian.")
        .def_prop_ro("num_inputs", &Local::num_inputs)
        .def_prop_ro("num_outputs", &Local::num_outputs)
        .def_prop_ro("has_jacobian", &Local::has_jacobian)
        .def("eval_value", &Local::eval_value, "xi"_a)
        .def("eval_jacobian", &Local::eval_jacobian, "xi"_a)
        .def(
            "accumulate",
            [](const Local& local, const Coordinate& xi, double weight, bool jacobian) {
                Eigen::VectorXd value = Eigen::VectorXd::Zero(local.num_outputs());
                Eigen::MatrixXd J =
                    Eigen::MatrixXd::Zero(jacobian ? local.num_outputs() : 0, jacobian ? local.num_inputs() : 0);
                local.accumulate(xi, weight, value, jacobian ? &J : nullptr);
                return nb::make_tuple(value, jacobian ? nb::cast(J) : nb::none());
            },
            "xi"_a, "weight"_a, "jacobian"_a = false,
            "Weighted (value, Jacobian or None); C++ adds into caller buffers. Geometry is not applied here.");
}
} // namespace

void bind_fem_local_local_response(nb::module_& m) {
    bind_local_response<2>(m, "LocalResponse2D");
    bind_local_response<3>(m, "LocalResponse3D");
}
} // namespace litepgo::fem::binding
