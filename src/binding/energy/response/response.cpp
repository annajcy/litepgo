#include "litepgo/energy/response/response.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::energy::binding {
void bind_response(nb::module_& m) {
    nb::class_<ResponseEvaluation>(m, "ResponseEvaluation")
        .def_prop_ro(
            "value", [](ResponseEvaluation& e) { return Vec(e.value()); }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("num_inputs", &ResponseEvaluation::num_inputs)
        .def_prop_ro("num_outputs", &ResponseEvaluation::num_outputs)
        .def_prop_ro("has_hessian", &ResponseEvaluation::has_hessian)
        .def("jacobian", &ResponseEvaluation::jacobian, nb::call_guard<nb::gil_scoped_release>())
        .def("hessian", &ResponseEvaluation::hessian, "weights"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("jvp", &ResponseEvaluation::jvp, "direction"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("vjp", &ResponseEvaluation::vjp, "weights"_a, nb::call_guard<nb::gil_scoped_release>());
    nb::class_<Response>(m, "Response", nb::is_weak_referenceable())
        .def_prop_ro("num_inputs", &Response::num_inputs)
        .def_prop_ro("num_outputs", &Response::num_outputs)
        .def_prop_ro("has_hessian", &Response::has_hessian)
        .def("evaluate", &Response::evaluate, "x"_a, nb::keep_alive<0, 1>(), nb::call_guard<nb::gil_scoped_release>());
}
} // namespace litepgo::energy::binding
