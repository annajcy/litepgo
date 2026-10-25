#include "litepgo/bindings/bindings.hpp"
#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/solver/newton.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::bindings {

void bind_solver(nb::module_& m) {
    m.doc() = "Nonlinear solvers (cpp/litepgo/solver)";

    nb::enum_<NewtonStatus>(m, "NewtonStatus")
        .value("Converged", NewtonStatus::Converged)
        .value("MaxIterations", NewtonStatus::MaxIterations)
        .value("StepTooSmall", NewtonStatus::StepTooSmall)
        .value("NoDescentDirection", NewtonStatus::NoDescentDirection);

    nb::class_<NewtonOptions>(m, "NewtonOptions")
        .def(
            "__init__",
            [](NewtonOptions* self, double gradient_tolerance, int max_iterations, HessianKind hessian,
               int max_line_search_steps, double line_search_shrink) {
                new (self) NewtonOptions{gradient_tolerance, max_iterations, hessian, max_line_search_steps,
                                         line_search_shrink};
            },
            "gradient_tolerance"_a, "max_iterations"_a = 50, "hessian"_a = HessianKind::Projected,
            "max_line_search_steps"_a = 30, "line_search_shrink"_a = 0.5)
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

    // The solve runs without the GIL; the Python callback takes it back per call.
    // It receives each NewtonStep by value, so Python may keep it after the call.
    m.def(
        "newton_solve",
        [](Energy& energy, const ConstVecRef& x0, const NewtonOptions& options,
           std::optional<std::vector<int>> fixed_dofs, SparseSolver* sparse_solver,
           std::optional<std::function<void(NewtonStep)>> on_iteration) {
            const DofReduction fixed(energy.num_dofs(), fixed_dofs.value_or(std::vector<int>{}));
            std::function<void(const NewtonStep&)> callback;
            if (on_iteration) {
                callback = [&on_iteration](const NewtonStep& step) {
                    nb::gil_scoped_acquire acquire;
                    (*on_iteration)(step);
                };
            }
            nb::gil_scoped_release release;
            if (sparse_solver != nullptr) {
                return newton_solve(energy, fixed, x0, options, *sparse_solver, callback);
            }
            AutoSparseSolver default_solver;
            return newton_solve(energy, fixed, x0, options, default_solver, callback);
        },
        "energy"_a, "x0"_a, "options"_a, "fixed_dofs"_a = nb::none(), "sparse_solver"_a = nb::none(),
        "on_iteration"_a = nb::none(),
        "Minimize energy from x0 with Newton's method; fixed_dofs keep their x0 values. sparse_solver defaults to "
        "AutoSparseSolver(); pass AutoIndefiniteSparseSolver() with HessianKind.Exact.");
}

} // namespace litepgo::bindings
