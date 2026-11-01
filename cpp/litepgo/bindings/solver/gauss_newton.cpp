#include "litepgo/solver/gauss_newton.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/solver/newton.hpp"

#include <nanobind/eigen/dense.h>

#include <limits>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::solver::binding {
void bind_gauss_newton(nb::module_& m) {

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
        [](energy::Response& residual, const ConstVecRef& initial, const GaussNewtonOptions& options,
           linalg::SparseSolver* linear_solver) {
            nb::gil_scoped_release release;
            if (linear_solver)
                return gauss_newton_solve(residual, initial, options, *linear_solver);
            return gauss_newton_solve(residual, initial, options);
        },
        "residual"_a, "initial"_a, "options"_a, "solver"_a = nb::none(),
        "Minimize 0.5*||response(x)||^2 using sparse damped normal equations. Defaults to AutoSparseSolver.");
}
} // namespace litepgo::solver::binding
