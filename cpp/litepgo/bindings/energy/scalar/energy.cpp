#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/bindings/bindings.hpp"

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
        .value("Projected", HessianKind::Projected, "A PSD approximation assembled from per-element projections.");

    // Energies are held by std::shared_ptr, so composites keep their Python
    // children alive. The GIL is released while C++ evaluates.
    nb::class_<Energy>(m, "Energy")
        .def_prop_ro("num_dofs", &Energy::num_dofs)
        .def_prop_ro("has_fixed_pattern", &Energy::has_fixed_pattern)
        .def_prop_ro(
            "hessian_pattern", [](const Energy& energy) { return energy.hessian_pattern(); },
            "The current Hessian sparsity pattern (zero values), as a copy.")
        .def("update_pattern", &Energy::update_pattern, "x"_a, nb::call_guard<nb::gil_scoped_release>(),
             "Adapt the Hessian pattern to x; returns whether it changed.")
        .def("eval_value", &Energy::eval_value, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_gradient", &Energy::eval_gradient, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "eval_hessian",
            [](const Energy& energy, const ConstVecRef& x) { return energy.eval_hessian(x, HessianKind::Exact); },
            "x"_a, nb::call_guard<nb::gil_scoped_release>(), "The exact Hessian.")
        .def(
            "eval_projected_hessian",
            [](const Energy& energy, const ConstVecRef& x) { return energy.eval_hessian(x, HessianKind::Projected); },
            "x"_a, nb::call_guard<nb::gil_scoped_release>(), "The PSD-projected Hessian.")
        .def(
            "eval",
            [](const Energy& energy, const ConstVecRef& x, bool want_gradient, std::optional<HessianKind> kind) {
                Vec gradient;
                SpMat hessian;
                double value = 0.0;
                {
                    nb::gil_scoped_release release;
                    if (kind) {
                        hessian = energy.hessian_pattern();
                    }
                    value = energy.eval(x, want_gradient ? &gradient : nullptr, kind ? &hessian : nullptr,
                                        kind.value_or(HessianKind::Exact));
                }
                return nb::make_tuple(value, want_gradient ? nb::cast(std::move(gradient)) : nb::none(),
                                      kind ? nb::cast(std::move(hessian)) : nb::none());
            },
            "x"_a, "gradient"_a = false, "hessian"_a = nb::none(),
            "One assembly returning (value, gradient or None, hessian or None); pass a HessianKind as hessian.")
        .def("max_feasible_step", &Energy::max_feasible_step, "x"_a, "dx"_a, nb::call_guard<nb::gil_scoped_release>(),
             "Largest alpha in [0, 1] keeping x + t dx feasible for all t in [0, alpha].");
}
} // namespace litepgo::energy::binding
