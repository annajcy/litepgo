#include "litepgo/bindings/bindings.hpp"
#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include "litepgo/solver/newton.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::solver::binding {

void bind(nb::module_& m) {
    m.doc() = "Nonlinear solvers (cpp/litepgo/solver)";

    nb::enum_<GaussNewtonStatus>(m, "GaussNewtonStatus")
        .value("Converged", GaussNewtonStatus::Converged)
        .value("MaxIterations", GaussNewtonStatus::MaxIterations)
        .value("StepTooSmall", GaussNewtonStatus::StepTooSmall);
    nb::class_<GaussNewtonOptions>(m, "GaussNewtonOptions")
        .def(
            "__init__",
            [](GaussNewtonOptions* self, double tolerance, int iterations, int trials, double damping) {
                new (self) GaussNewtonOptions{tolerance, iterations, trials, damping};
            },
            "gradient_tolerance"_a = std::numeric_limits<double>::quiet_NaN(), "max_iterations"_a = 50,
            "max_trial_steps"_a = 20, "initial_damping"_a = 1e-3)
        .def_rw("gradient_tolerance", &GaussNewtonOptions::gradient_tolerance)
        .def_rw("max_iterations", &GaussNewtonOptions::max_iterations)
        .def_rw("max_trial_steps", &GaussNewtonOptions::max_trial_steps)
        .def_rw("initial_damping", &GaussNewtonOptions::initial_damping);
    nb::class_<GaussNewtonResult>(m, "GaussNewtonResult")
        .def_ro("x", &GaussNewtonResult::x)
        .def_ro("objective", &GaussNewtonResult::objective)
        .def_ro("gradient_norm", &GaussNewtonResult::gradient_norm)
        .def_ro("iterations", &GaussNewtonResult::iterations)
        .def_ro("evaluations", &GaussNewtonResult::evaluations)
        .def_ro("status", &GaussNewtonResult::status);
    m.def(
        "gauss_newton_solve",
        [](nb::callable callback, const ConstVecRef& initial, const GaussNewtonOptions& options) {
            ResidualFunction residual = [&](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
                nb::gil_scoped_acquire acquire;
                // Own the argument so callbacks may safely retain it.
                nb::object value = callback(nb::cast(Vec(x)), J != nullptr);
                if (value.is_none())
                    throw std::domain_error("Residual is infeasible");
                if (!J)
                    return nb::cast<Vec>(value);
                if (!nb::isinstance<nb::tuple>(value) || nb::len(value) != 2)
                    throw nb::type_error("Expected (residual, Jacobian) when need_jacobian is True");
                nb::tuple pair = nb::cast<nb::tuple>(value);
                *J = nb::cast<Eigen::MatrixXd>(pair[1]);
                return nb::cast<Vec>(pair[0]);
            };
            nb::gil_scoped_release release;
            return gauss_newton_solve(residual, initial, options);
        },
        "residual"_a, "initial"_a, "options"_a,
        "Minimize 0.5*||r||^2. residual(x, need_jacobian) returns (r, J) when requested, "
        "otherwise r. Return None for an infeasible point; Python exceptions propagate. No finite differences.");

    nb::enum_<NewtonStatus>(m, "NewtonStatus")
        .value("Converged", NewtonStatus::Converged)
        .value("MaxIterations", NewtonStatus::MaxIterations)
        .value("StepTooSmall", NewtonStatus::StepTooSmall)
        .value("NoDescentDirection", NewtonStatus::NoDescentDirection);

    nb::class_<NewtonOptions>(m, "NewtonOptions")
        .def(
            "__init__",
            [](NewtonOptions* self, double gradient_tolerance, int max_iterations, energy::HessianKind hessian,
               int max_line_search_steps, double line_search_shrink) {
                new (self) NewtonOptions{gradient_tolerance, max_iterations, hessian, max_line_search_steps,
                                         line_search_shrink};
            },
            "gradient_tolerance"_a = std::numeric_limits<double>::quiet_NaN(), "max_iterations"_a = 50,
            "hessian"_a = energy::HessianKind::Projected, "max_line_search_steps"_a = 30, "line_search_shrink"_a = 0.5)
        .def_rw("gradient_tolerance", &NewtonOptions::gradient_tolerance)
        .def_rw("max_iterations", &NewtonOptions::max_iterations)
        .def_rw("hessian", &NewtonOptions::hessian)
        .def_rw("max_line_search_steps", &NewtonOptions::max_line_search_steps)
        .def_rw("line_search_shrink", &NewtonOptions::line_search_shrink);

    nb::class_<NewtonStep>(m, "NewtonStep")
        .def_ro("iteration", &NewtonStep::iteration)
        .def_ro("energy", &NewtonStep::energy)
        .def_ro("gradient_norm", &NewtonStep::gradient_norm)
        .def_ro("pattern_changed", &NewtonStep::pattern_changed)
        .def_ro("diagonal_shifts", &NewtonStep::diagonal_shifts)
        .def_ro("feasible_step", &NewtonStep::feasible_step)
        .def_ro("step", &NewtonStep::step)
        .def_ro("line_search_evaluations", &NewtonStep::line_search_evaluations);

    nb::class_<NewtonResult>(m, "NewtonResult")
        .def_ro("status", &NewtonResult::status)
        .def_ro("x", &NewtonResult::x)
        .def_ro("energy", &NewtonResult::energy)
        .def_ro("gradient_norm", &NewtonResult::gradient_norm)
        .def_ro("iterations", &NewtonResult::iterations);

    // Mirrors the C++ signature; solver defaults to AutoSparseSolver(). The solve runs
    // without the GIL; the Python callback takes it back per call. It receives each
    // NewtonStep by value, so Python may keep it after the call.
    m.def(
        "newton_solve",
        [](energy::Energy& energy, const dof::DofReduction& fixed, const ConstVecRef& x0, const NewtonOptions& options,
           linalg::SparseSolver* solver, std::optional<std::function<void(NewtonStep)>> on_iteration) {
            std::function<void(const NewtonStep&)> callback;
            if (on_iteration) {
                callback = [&on_iteration](const NewtonStep& step) {
                    nb::gil_scoped_acquire acquire;
                    (*on_iteration)(step);
                };
            }
            nb::gil_scoped_release release;
            if (solver != nullptr) {
                return newton_solve(energy, fixed, x0, options, *solver, callback);
            }
            linalg::AutoSparseSolver default_solver;
            return newton_solve(energy, fixed, x0, options, default_solver, callback);
        },
        "energy"_a, "fixed"_a, "x0"_a, "options"_a, "solver"_a = nb::none(), "on_iteration"_a = nb::none(),
        "Minimize energy from x0 with Newton's method; fixed DOFs keep their x0 values. Pass "
        "AutoIndefiniteSparseSolver() with HessianKind.Exact.");
}

} // namespace litepgo::solver::binding
