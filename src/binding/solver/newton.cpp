#include "litepgo/solver/newton.hpp"
#include "bindings.hpp"
#include "convergence.hpp"
#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>

#include <functional>
#include <limits>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::solver::binding {
void bind_newton(nb::module_& m) {
    nb::enum_<NewtonStatus>(m, "NewtonStatus")
        .value("Converged", NewtonStatus::Converged)
        .value("MaxIterations", NewtonStatus::MaxIterations)
        .value("StepTooSmall", NewtonStatus::StepTooSmall)
        .value("NoDescentDirection", NewtonStatus::NoDescentDirection)
        .value("PrecisionStagnation", NewtonStatus::PrecisionStagnation);

    nb::class_<NewtonConvergenceState>(m, "ConvergenceState")
        .def_ro("iteration", &NewtonConvergenceState::iteration)
        .def_ro("energy", &NewtonConvergenceState::energy)
        .def_ro("gradient_norm", &NewtonConvergenceState::gradient_norm)
        .def_ro("project_hessian", &NewtonConvergenceState::project_hessian)
        .def_ro("newton_correction_norm", &NewtonConvergenceState::newton_correction_norm)
        .def_ro("diagonal_shifts", &NewtonConvergenceState::diagonal_shifts)
        .def_ro("linear_backward_error", &NewtonConvergenceState::linear_backward_error);
    nb::class_<AbsoluteConvergence>(m, "AbsoluteConvergence")
        .def(nb::init<double>(), "tolerance"_a)
        .def_prop_ro("tolerance", &AbsoluteConvergence::tolerance)
        .def("__call__", &AbsoluteConvergence::operator());

    nb::class_<NewtonOptions>(m, "NewtonOptions")
        .def(
            "__init__",
            [](NewtonOptions* self, int max_iterations, bool project_hessian, int max_line_search_steps,
               double line_search_shrink, double max_coordinate_step, bool detect_precision_stagnation,
               double precision_coordinate_scale) {
                new (self) NewtonOptions{max_iterations,
                                         project_hessian,
                                         max_line_search_steps,
                                         line_search_shrink,
                                         max_coordinate_step,
                                         detect_precision_stagnation,
                                         precision_coordinate_scale};
            },
            "max_iterations"_a = 50, "project_hessian"_a = true, "max_line_search_steps"_a = 30,
            "line_search_shrink"_a = 0.5, "max_coordinate_step"_a = std::numeric_limits<double>::infinity(),
            "detect_precision_stagnation"_a = false, "precision_coordinate_scale"_a = 1.0)
        .def_rw("max_iterations", &NewtonOptions::max_iterations)
        .def_rw("project_hessian", &NewtonOptions::project_hessian)
        .def_rw("max_line_search_steps", &NewtonOptions::max_line_search_steps)
        .def_rw("line_search_shrink", &NewtonOptions::line_search_shrink)
        .def_rw("max_coordinate_step", &NewtonOptions::max_coordinate_step)
        .def_rw("detect_precision_stagnation", &NewtonOptions::detect_precision_stagnation)
        .def_rw("precision_coordinate_scale", &NewtonOptions::precision_coordinate_scale);

    nb::class_<NewtonStep>(m, "NewtonStep")
        .def_ro("iteration", &NewtonStep::iteration)
        .def_ro("energy", &NewtonStep::energy)
        .def_ro("gradient_norm", &NewtonStep::gradient_norm)
        .def_ro("pattern_changed", &NewtonStep::pattern_changed)
        .def_ro("diagonal_shifts", &NewtonStep::diagonal_shifts)
        .def_ro("feasible_step", &NewtonStep::feasible_step)
        .def_ro("step", &NewtonStep::step)
        .def_ro("line_search_evaluations", &NewtonStep::line_search_evaluations)
        .def_ro("scaled_direction_norm", &NewtonStep::scaled_direction_norm)
        .def_ro("linear_backward_error", &NewtonStep::linear_backward_error)
        .def_ro("precision_stagnation_steps", &NewtonStep::precision_stagnation_steps);

    nb::class_<NewtonResult>(m, "NewtonResult")
        .def_ro("status", &NewtonResult::status)
        .def_ro("x", &NewtonResult::x)
        .def_ro("energy", &NewtonResult::energy)
        .def_ro("gradient_norm", &NewtonResult::gradient_norm)
        .def_ro("iterations", &NewtonResult::iterations)
        .def_ro("scaled_direction_norm", &NewtonResult::scaled_direction_norm)
        .def_ro("linear_backward_error", &NewtonResult::linear_backward_error)
        .def_ro("precision_stagnation_steps", &NewtonResult::precision_stagnation_steps);

    // Mirrors the C++ signature; solver defaults to AutoSparseSolver(). The solve runs
    // without the GIL; the Python callback takes it back per call. It receives each
    // NewtonStep by value, so Python may keep it after the call.
    m.def(
        "newton_solve",
        [](energy::Energy& energy, const dof::DofReduction& fixed, const ConstVecRef& x0, const NewtonOptions& options,
           linalg::SparseSolver* solver, std::optional<std::function<void(NewtonStep)>> on_iteration,
           nb::object convergence) {
            const auto check = convergence_check(convergence);
            std::function<void(const NewtonStep&)> callback;
            if (on_iteration) {
                callback = [&on_iteration](const NewtonStep& step) {
                    nb::gil_scoped_acquire acquire;
                    (*on_iteration)(step);
                };
            }
            nb::gil_scoped_release release;
            if (solver != nullptr) {
                return newton_solve(energy, fixed, x0, options, *solver, check, callback);
            }
            linalg::AutoSparseSolver default_solver;
            return newton_solve(energy, fixed, x0, options, default_solver, check, callback);
        },
        "energy"_a, "fixed"_a, "x0"_a, "options"_a, "solver"_a = nb::none(), "on_iteration"_a = nb::none(),
        nb::kw_only(), "convergence"_a,
        "Minimize energy from x0 with Newton's method; fixed DOFs keep their x0 values. Pass "
        "AutoIndefiniteSparseSolver() with project_hessian=False.");
}
} // namespace litepgo::solver::binding
