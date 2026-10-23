#include "litepgo/energy/energy.h"
#include "litepgo/bindings/bindings.h"
#include "litepgo/bindings/sparse_matrix.h"
#include "litepgo/energy/affine_mapped_energy.h"
#include "litepgo/energy/quadratic_energy.h"
#include "litepgo/energy/weighted_energy_sum.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::bindings {

void bind_energy(nb::module_& m) {
    m.doc() = "Energies and their composites (cpp/litepgo/energy)";

    // InfeasibleState: evaluating at an infeasible state, recoverable by a shorter step.
    nb::exception<InfeasibleState>(m, "InfeasibleStateError", PyExc_ValueError);

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
        .def("value", &Energy::value, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("gradient", &Energy::gradient, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "hessian", [](const Energy& energy, const ConstVecRef& x) { return energy.hessian(x, HessianKind::Exact); },
            "x"_a, nb::call_guard<nb::gil_scoped_release>(), "The exact Hessian.")
        .def(
            "projected_hessian",
            [](const Energy& energy, const ConstVecRef& x) { return energy.hessian(x, HessianKind::Projected); }, "x"_a,
            nb::call_guard<nb::gil_scoped_release>(), "The PSD-projected Hessian.")
        .def(
            "evaluate",
            [](const Energy& energy, const ConstVecRef& x, bool want_gradient, std::optional<HessianKind> kind) {
                Vec gradient;
                SpMat hessian;
                double value = 0.0;
                {
                    nb::gil_scoped_release release;
                    if (kind) {
                        hessian = energy.hessian_pattern();
                    }
                    value = energy.evaluate(x, want_gradient ? &gradient : nullptr, kind ? &hessian : nullptr,
                                            kind.value_or(HessianKind::Exact));
                }
                return nb::make_tuple(value, want_gradient ? nb::cast(std::move(gradient)) : nb::none(),
                                      kind ? nb::cast(std::move(hessian)) : nb::none());
            },
            "x"_a, "gradient"_a = false, "hessian"_a = nb::none(),
            "One assembly returning (value, gradient or None, hessian or None); pass a HessianKind as hessian.")
        .def("max_feasible_step", &Energy::max_feasible_step, "x"_a, "dx"_a, nb::call_guard<nb::gil_scoped_release>(),
             "Largest alpha in [0, 1] keeping x + t dx feasible for all t in [0, alpha].");

    nb::class_<QuadraticEnergy, Energy>(m, "QuadraticEnergy")
        .def(
            "__init__",
            [](QuadraticEnergy* self, const SparseArgument& A, std::optional<Vec> b) {
                new (self) QuadraticEnergy(A.matrix, b.value_or(Vec()));
            },
            "A"_a, "b"_a = nb::none(), "E(x) = 1/2 x^T A x + b^T x with a symmetric A.");

    nb::class_<WeightedEnergySum, Energy>(m, "WeightedEnergySum")
        .def(
            "__init__",
            [](WeightedEnergySum* self, const std::vector<std::pair<std::shared_ptr<Energy>, double>>& terms) {
                std::vector<WeightedEnergy> weighted;
                for (const auto& [energy, weight] : terms) {
                    weighted.push_back({energy, weight});
                }
                new (self) WeightedEnergySum(std::move(weighted));
            },
            "terms"_a, "sum_i w_i E_i(x) from (energy, weight) pairs over the same DOFs; weights are nonzero.");

    nb::class_<AffineMappedEnergy, Energy>(m, "AffineMappedEnergy")
        .def(
            "__init__",
            [](AffineMappedEnergy* self, std::shared_ptr<Energy> energy, const SparseArgument& T,
               std::optional<Vec> offset) {
                new (self) AffineMappedEnergy(std::move(energy), T.matrix, offset.value_or(Vec()));
            },
            "energy"_a, "T"_a, "offset"_a = nb::none(), "E(z) = energy(T z + offset).");
}

} // namespace litepgo::bindings
