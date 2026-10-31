#include "litepgo/energy/energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/sparse_matrix.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/affine_mapped_energy.hpp"
#include "litepgo/energy/batched_mesh_energy.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/energy/quadratic_energy.hpp"
#include "litepgo/energy/vertex_pulling_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>

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
    using BatchCallback = std::function<BatchResult(std::vector<int>, BatchMatrix, bool, bool)>;
    using BatchStep = std::function<Vec(std::vector<int>, BatchMatrix, BatchMatrix)>;
    nb::class_<BatchedMeshEnergy, Energy>(m, "BatchedMeshEnergy")
        .def(
            "__init__",
            [](BatchedMeshEnergy* self, const dof::StencilDofMap& map, BatchCallback callback, int batch_size,
               std::optional<BatchStep> step) {
                if (!callback || (step && !*step))
                    throw std::invalid_argument("Expected batch callbacks");
                auto evaluate = [callback](std::span<const int> ids, const BatchMatrix& x, BatchMatrix* g,
                                           BatchMatrix* h) {
                    auto [values, gradients, hessians] =
                        callback(std::vector<int>(ids.begin(), ids.end()), x, g != nullptr, h != nullptr);
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
                BatchedMeshEnergy::Step bound;
                if (step)
                    bound = [step = *step](std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& dx) {
                        return step(std::vector<int>(ids.begin(), ids.end()), x, dx);
                    };
                new (self) BatchedMeshEnergy(map, std::move(evaluate), batch_size, std::move(bound));
            },
            "map"_a, "eval_batch"_a, "batch_size"_a = 256, "step"_a = nb::none());

    using LocalResult = std::tuple<double, std::optional<Vec>, std::optional<Eigen::MatrixXd>>;
    using LocalCallback = std::function<LocalResult(int, Eigen::MatrixXd, bool, bool)>;
    using StepCallback = std::function<double(int, Eigen::MatrixXd, Eigen::MatrixXd)>;
    nb::class_<MeshEnergy, Energy>(m, "MeshEnergy")
        .def(
            "__init__",
            [](MeshEnergy* self, const dof::StencilDofMap& map, LocalCallback callback,
               std::optional<StepCallback> step) {
                if (!callback || (step && !*step))
                    throw std::invalid_argument("MeshEnergy requires callable callbacks");
                auto local = [callback = std::move(callback)](int e, const Eigen::MatrixXd& x, Vec* g,
                                                              Eigen::MatrixXd* h) {
                    // Value arguments own their storage; Python may retain them safely.
                    auto [value, gradient, hessian] = callback(e, x, g != nullptr, h != nullptr);
                    if (g) {
                        if (!gradient)
                            throw std::invalid_argument("MeshEnergy callback omitted requested gradient");
                        *g = std::move(*gradient);
                    }
                    if (h) {
                        if (!hessian)
                            throw std::invalid_argument("MeshEnergy callback omitted requested Hessian");
                        *h = std::move(*hessian);
                    }
                    return value;
                };
                if (step) {
                    new (self)
                        MeshEnergy(map, std::move(local),
                                   [step = std::move(*step)](int e, const Eigen::MatrixXd& x,
                                                             const Eigen::MatrixXd& dx) { return step(e, x, dx); });
                } else {
                    new (self) MeshEnergy(map, std::move(local));
                }
            },
            "map"_a, "eval_local"_a, "step"_a = nb::none(),
            "Assemble fixed vertex stencils. eval_local(e, x_local, need_gradient, need_hessian) returns "
            "(value, gradient or None, Hessian or None). x_local has one row per vertex; derivatives use "
            "vertex-major scalar order. Callbacks may run on worker threads (with the GIL). "
            "Optional step(e, x_local, dx_local) returns a safe bound in [0,1]; omit only for energies "
            "feasible along every finite segment. Captured data must remain unchanged during evaluation.");

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
