#include "litepgo/solver/static_equilibrium_response.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/shared_ptr.h>
#include <utility>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::solver::binding {
struct StaticEquilibriumGC {
    static int traverse(PyObject* self, visitproc visit, void* arg) {
        Py_VISIT(Py_TYPE(self));
        if (!nb::inst_ready(self))
            return 0;
        const auto* response = nb::inst_ptr<StaticEquilibriumResponse>(self);
        nb::handle model = nb::find(response->model_.get());
        Py_VISIT(model.ptr());
        return 0;
    }
    static int clear(PyObject* self) {
        if (!nb::inst_ready(self))
            return 0;
        auto* response = nb::inst_ptr<StaticEquilibriumResponse>(self);
        auto model = std::exchange(response->model_, {});
        return 0;
    }
};
void bind_static_equilibrium_response(nb::module_& m) {
    static PyType_Slot slots[] = {{Py_tp_traverse, reinterpret_cast<void*>(StaticEquilibriumGC::traverse)},
                                  {Py_tp_clear, reinterpret_cast<void*>(StaticEquilibriumGC::clear)},
                                  {0, nullptr}};
    nb::class_<StaticEquilibriumResponse, energy::Response>(m, "StaticEquilibriumResponse", nb::type_slots(slots))
        .def(nb::init<std::shared_ptr<energy::ParametricEnergy>, dof::DofReduction, Vec, NewtonOptions>(), "model"_a,
             "fixed"_a, "initial"_a, "options"_a);
}
} // namespace litepgo::solver::binding
