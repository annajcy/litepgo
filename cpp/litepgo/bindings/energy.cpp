#include "litepgo/energy/energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/sparse_matrix.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/affine_mapped_energy.hpp"
#include "litepgo/energy/dynamic_pattern_energy.hpp"
#include "litepgo/energy/fixed_pattern_energy.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/energy/quadratic_energy.hpp"
#include "litepgo/energy/vertex_pulling_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/trampoline.h>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <array>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {

// Both bases share scalar evaluation and feasibility; only the dynamic base calls pattern_at.
template <class Base> class PyEnergy : public Base {
  public:
    NB_TRAMPOLINE(Base);
    PyEnergy(int n, const linalg::binding::SparseArgument& pattern) : Base(n, pattern.matrix) {}

  private:
    double eval_values(const ConstVecRef& x, Vec* g, SpMat* H, HessianKind kind) const override {
        nb::detail::ticket ticket(this->nb_trampoline, "eval_values", nb::detail::str_hash("eval_values"), true);
        nb::object out = this->nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x)), g != nullptr,
                                                                     H ? nb::cast(kind) : nb::none());
        if (out.is_none())
            throw std::domain_error("Energy is infeasible");
        if (!nb::isinstance<nb::tuple>(out) || nb::len(out) != 3)
            throw nb::type_error("Expected (value, gradient, Hessian)");
        nb::tuple values = nb::cast<nb::tuple>(out);
        if (g) {
            *g = nb::cast<Vec>(values[1]);
            if (g->size() != this->num_dofs() || !g->allFinite())
                throw std::logic_error("Energy: invalid gradient");
        }
        if (H) {
            const SpMat source = nb::cast<linalg::binding::SparseArgument>(values[2]).matrix;
            if (source.rows() != H->rows() || source.cols() != H->cols() || !source.coeffs().allFinite())
                throw std::logic_error("Energy: invalid Hessian");
            for (int c = 0; c < source.outerSize(); ++c)
                for (SpMat::InnerIterator it(source, c); it; ++it) {
                    bool found = false;
                    for (SpMat::InnerIterator slot(*H, c); slot; ++slot)
                        if (slot.row() == it.row()) {
                            found = true;
                            break;
                        }
                    if (!found && it.value() != 0)
                        throw std::logic_error("Energy: Hessian outside pattern; call update_pattern");
                }
            for (int c = 0; c < H->outerSize(); ++c)
                for (SpMat::InnerIterator it(*H, c); it; ++it)
                    it.valueRef() = source.coeff(it.row(), c);
        }
        return nb::cast<double>(values[0]);
    }
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override {
        nb::detail::ticket ticket(this->nb_trampoline, "feasible_step", nb::detail::str_hash("feasible_step"), false);
        if (!ticket.key.is_valid())
            return 1.0;
        return nb::cast<double>(this->nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x)), nb::cast(Vec(dx))));
    }
};

class PyDynamicEnergy : public PyEnergy<DynamicPatternEnergy> {
  public:
    using PyEnergy<DynamicPatternEnergy>::PyEnergy;

  private:
    SpMat pattern_at(const ConstVecRef& x) const override {
        nb::detail::ticket ticket(this->nb_trampoline, "pattern_at", nb::detail::str_hash("pattern_at"), true);
        return nb::cast<linalg::binding::SparseArgument>(this->nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x))))
            .matrix;
    }
};

void bind(nb::module_& m) {
    m.doc() = "Energies and their composites (cpp/litepgo/energy)";

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

    using BatchResult = std::tuple<Vec, std::optional<BatchMatrix>, std::optional<BatchMatrix>>;
    using BatchCallback = std::function<std::optional<BatchResult>(std::vector<int>, BatchMatrix, bool, bool)>;
    using BatchStep = std::function<Vec(std::vector<int>, BatchMatrix, BatchMatrix)>;
    nb::class_<FixedPatternEnergy, Energy, PyEnergy<FixedPatternEnergy>>(m, "FixedPatternEnergy")
        .def(nb::init<int, const linalg::binding::SparseArgument&>(), "num_dofs"_a, "pattern"_a)
        .def(
            "feasible_step", [](const FixedPatternEnergy&, const ConstVecRef&, const ConstVecRef&) { return 1.0; },
            "x"_a, "dx"_a);
    nb::class_<DynamicPatternEnergy, Energy, PyDynamicEnergy>(m, "DynamicPatternEnergy")
        .def(nb::init<int, const linalg::binding::SparseArgument&>(), "num_dofs"_a, "initial_pattern"_a)
        .def(
            "feasible_step", [](const DynamicPatternEnergy&, const ConstVecRef&, const ConstVecRef&) { return 1.0; },
            "x"_a, "dx"_a);

    auto mesh = nb::class_<MeshEnergy, Energy>(m, "MeshEnergy");
    auto bind_map = [&]<class Map>() {
        mesh.def(
            "__init__",
            [](MeshEnergy* self, const Map& map, BatchCallback callback, int batch_size,
               std::optional<BatchStep> step) {
                if (!callback || (step && !*step))
                    throw std::invalid_argument("Expected batch callbacks");
                auto evaluate = [callback](std::span<const int> ids, const BatchMatrix& x, BatchMatrix* g,
                                           BatchMatrix* h) {
                    auto result = callback(std::vector<int>(ids.begin(), ids.end()), x, g != nullptr, h != nullptr);
                    if (!result)
                        throw std::domain_error("MeshEnergy: infeasible batch");
                    auto [values, gradients, hessians] = std::move(*result);
                    if (g) {
                        if (!gradients)
                            throw std::invalid_argument("Missing batch gradients");
                        *g = std::move(*gradients);
                    }
                    if (h) {
                        if (!hessians)
                            throw std::invalid_argument("Missing batch Hessians");
                        *h = std::move(*hessians);
                    }
                    return values;
                };
                MeshEnergy::Step bound;
                if (step)
                    bound = [step = *step](std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& dx) {
                        return step(std::vector<int>(ids.begin(), ids.end()), x, dx);
                    };
                new (self) MeshEnergy(map, std::move(evaluate), batch_size, std::move(bound));
            },
            "map"_a, "eval_batch"_a, "batch_size"_a = 256, "step"_a = nb::none());
    };
    bind_map.template operator()<dof::StencilDofMap>();
    bind_map.template operator()<dof::ConstantDofMap>();
    bind_map.template operator()<dof::ElementDofMap>();
    bind_map.template operator()<dof::VertexDofMap>();

    nb::class_<QuadraticEnergy, Energy>(m, "QuadraticEnergy")
        .def(
            "__init__",
            [](QuadraticEnergy* self, const linalg::binding::SparseArgument& A, std::optional<Vec> b) {
                new (self) QuadraticEnergy(A.matrix, b.value_or(Vec()));
            },
            "A"_a, "b"_a = nb::none(), "E(x) = 1/2 x^T A x + b^T x with a symmetric A.");

    nb::class_<VertexPullingEnergy, Energy>(m, "VertexPullingEnergy")
        .def(
            "__init__",
            [](VertexPullingEnergy* self, int num_dofs,
               const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>& position_dofs,
               const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& targets, double stiffness) {
                std::vector<std::array<int, 3>> dofs(static_cast<std::size_t>(position_dofs.rows()));
                for (Eigen::Index i = 0; i < position_dofs.rows(); ++i)
                    for (int c = 0; c < 3; ++c)
                        dofs[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)] = position_dofs(i, c);
                if (targets.rows() != position_dofs.rows())
                    throw std::invalid_argument("VertexPullingEnergy requires one target row per pulled vertex");
                new (self)
                    VertexPullingEnergy(num_dofs, std::move(dofs), targets.reshaped<Eigen::RowMajor>(), stiffness);
            },
            "num_dofs"_a, "position_dofs"_a, "targets"_a, "stiffness"_a,
            "E(x) = k/2 sum_i |x[position_dofs[i]] - targets[i]|^2 from (n, 3) DOF and target arrays.");

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
            [](AffineMappedEnergy* self, std::shared_ptr<Energy> energy, const linalg::binding::SparseArgument& T,
               std::optional<Vec> offset) {
                new (self) AffineMappedEnergy(std::move(energy), T.matrix, offset.value_or(Vec()));
            },
            "energy"_a, "T"_a, "offset"_a = nb::none(), "E(z) = energy(T z + offset).");
}

} // namespace litepgo::energy::binding
