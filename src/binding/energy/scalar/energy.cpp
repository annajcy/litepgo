#include "litepgo/energy/scalar/energy.hpp"
#include "bindings.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>
#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_energy(nb::module_& m) {

    nb::enum_<HessianKind>(m, "HessianKind")
        .value("Exact", HessianKind::Exact, "The exact Hessian.")
        .value("Projected", HessianKind::Projected,
               "An implementation-provided solver approximation; not necessarily positive semidefinite.");

    // Energies are held by std::shared_ptr, so composites keep their Python
    // children alive. The GIL is released while C++ evaluates.
    nb::class_<Energy>(m, "Energy")
        .def_prop_ro("num_dofs", &Energy::num_dofs)
        .def("eval_value", &Energy::eval_value, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_gradient", &Energy::eval_gradient, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_hessian", &Energy::eval_hessian, "x"_a, "kind"_a = HessianKind::Exact,
             nb::call_guard<nb::gil_scoped_release>())
        .def(
            "eval",
            [](const Energy& energy, const ConstVecRef& x, bool want_gradient, std::optional<HessianKind> kind) {
                EnergyEvaluation result;
                {
                    nb::gil_scoped_release release;
                    result = energy.eval(x, want_gradient, kind);
                }
                return nb::make_tuple(result.value,
                                      result.gradient ? nb::cast(std::move(*result.gradient)) : nb::none(),
                                      result.hessian ? nb::cast(std::move(*result.hessian)) : nb::none());
            },
            "x"_a, "gradient"_a = false, "hessian"_a = nb::none(),
            "One assembly returning (value, gradient or None, hessian or None); pass a HessianKind as hessian.")
        .def("max_feasible_step", &Energy::max_feasible_step, "x"_a, "dx"_a, nb::call_guard<nb::gil_scoped_release>(),
             "A conservative alpha in [0, 1] keeping x + t dx feasible for all t in [0, alpha]; not necessarily the "
             "largest.");
}
} // namespace litepgo::energy::binding
