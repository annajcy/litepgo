#include "litepgo/energy/response/response.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_response(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<Response>(m, "Response", nb::is_weak_referenceable())
        .def_prop_ro("num_inputs", &Response::num_inputs)
        .def_prop_ro("num_outputs", &Response::num_outputs)
        .def_prop_ro("has_hessian", &Response::has_hessian)
        .def("eval_hessian", &Response::eval_hessian, "x"_a, "weights"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_value", &Response::eval_value, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_jacobian", &Response::eval_jacobian, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_jvp", &Response::eval_jvp, "x"_a, "dx"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_vjp", &Response::eval_vjp, "x"_a, "weights"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_value_and_jacobian", &Response::eval_value_and_jacobian, "x"_a,
             nb::call_guard<nb::gil_scoped_release>());
}
} // namespace litepgo::energy::binding
