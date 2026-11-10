#include "litepgo/energy/scalar/energy.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::energy::binding {
void bind_energy(nb::module_& m) {
    nb::class_<EnergyEvaluation>(m, "EnergyEvaluation")
        .def_prop_ro("num_dofs", &EnergyEvaluation::num_dofs)
        .def_prop_ro("value", &EnergyEvaluation::value, nb::call_guard<nb::gil_scoped_release>())
        .def("gradient", &EnergyEvaluation::gradient, nb::call_guard<nb::gil_scoped_release>())
        .def("hessian", &EnergyEvaluation::hessian, nb::call_guard<nb::gil_scoped_release>())
        .def("projected_hessian", &EnergyEvaluation::projected_hessian, nb::call_guard<nb::gil_scoped_release>())
        .def("max_feasible_step", &EnergyEvaluation::max_feasible_step, "direction"_a,
             nb::call_guard<nb::gil_scoped_release>());
    nb::class_<Energy>(m, "Energy", nb::is_weak_referenceable())
        .def_prop_ro("num_dofs", &Energy::num_dofs)
        .def("evaluate", &Energy::evaluate, "x"_a, nb::keep_alive<0, 1>(), nb::call_guard<nb::gil_scoped_release>());
}
} // namespace litepgo::energy::binding
