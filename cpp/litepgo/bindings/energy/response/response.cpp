#include "litepgo/energy/response/response.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/energy/response/static_equilibrium_response.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <utility>
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
struct StaticEquilibriumGC {
    static int traverse(PyObject* self, visitproc visit, void* arg) {
        Py_VISIT(Py_TYPE(self));
        if (!nb::inst_ready(self))
            return 0;
        const auto* response = nb::inst_ptr<StaticEquilibriumResponse>(self);
        nb::handle make_energy = nb::find(response->make_energy_);
        nb::handle partial = nb::find(response->partial_);
        Py_VISIT(make_energy.ptr());
        Py_VISIT(partial.ptr());
        return 0;
    }
    static int clear(PyObject* self) {
        if (!nb::inst_ready(self))
            return 0;
        auto* response = nb::inst_ptr<StaticEquilibriumResponse>(self);
        // Empty both members before releasing callbacks, which may run Python finalizers.
        auto make_energy = std::exchange(response->make_energy_, {});
        auto partial = std::exchange(response->partial_, {});
        return 0;
    }
};
void bind_static_equilibrium_response(nb::module_& m) {
    static PyType_Slot slots[] = {{Py_tp_traverse, reinterpret_cast<void*>(StaticEquilibriumGC::traverse)},
                                  {Py_tp_clear, reinterpret_cast<void*>(StaticEquilibriumGC::clear)},
                                  {0, nullptr}};
    nb::class_<StaticEquilibriumResponse, Response>(m, "StaticEquilibriumResponse", nb::type_slots(slots))
        .def(nb::init<int, dof::DofReduction, Vec, solver::NewtonOptions, StaticEquilibriumResponse::MakeEnergy,
                      StaticEquilibriumResponse::Partial>(),
             "num_parameters"_a, "fixed"_a, "initial"_a, "options"_a, "make_energy"_a, "partial"_a);
}
} // namespace litepgo::energy::binding
