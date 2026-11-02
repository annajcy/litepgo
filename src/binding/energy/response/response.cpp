#include "litepgo/energy/response/response.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_response(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<Response>(m, "Response")
        .def_prop_ro("num_inputs", &Response::num_inputs)
        .def_prop_ro("num_outputs", &Response::num_outputs)
        .def_prop_ro("has_fixed_pattern", &Response::has_fixed_pattern)
        .def_prop_ro("jacobian_pattern", [](const Response& r) { return r.jacobian_pattern(); })
        .def("update_pattern", &Response::update_pattern, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_value", &Response::eval_value, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_jacobian", &Response::eval_jacobian, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "eval",
            [](const Response& r, const ConstVecRef& x, bool need_jacobian) {
                SpMat J = r.jacobian_pattern();
                Vec value;
                {
                    nb::gil_scoped_release release;
                    value = r.eval(x, need_jacobian ? &J : nullptr);
                }
                return nb::make_tuple(value, need_jacobian ? nb::cast(J) : nb::none());
            },
            "x"_a, "need_jacobian"_a = false);
}
} // namespace litepgo::energy::binding
