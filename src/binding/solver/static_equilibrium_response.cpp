#include "litepgo/solver/static_equilibrium_response.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/function.h>
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
    nb::class_<StaticEquilibriumResponse, energy::Response>(m, "StaticEquilibriumResponse", nb::type_slots(slots))
        .def(nb::init<int, dof::DofReduction, Vec, NewtonOptions, StaticEquilibriumResponse::MakeEnergy,
                      StaticEquilibriumResponse::Partial>(),
             "num_parameters"_a, "fixed"_a, "initial"_a, "options"_a, "make_energy"_a, "partial"_a);
}
} // namespace litepgo::solver::binding
