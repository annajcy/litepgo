#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#include <functional>
#include <limits>

namespace litepgo {

struct NewtonOptions {
    // Required: converged when the free DOFs' gradient max-norm is at most this.
    double gradient_tolerance = std::numeric_limits<double>::quiet_NaN();
    int max_iterations = 50;
    HessianKind hessian = HessianKind::Projected;
    int max_line_search_steps = 30;
    double line_search_shrink = 0.5;
};

enum class NewtonStatus {
    // The free gradient max-norm is at most gradient_tolerance.
    Converged,
    MaxIterations,
    // The line search found no acceptable point within max_line_search_steps.
    StepTooSmall,
    // Even with the largest diagonal shift, the system would not factor or its
    // solution was not a descent direction.
    NoDescentDirection,
};

// What one iteration did, as reported to the callback.
struct NewtonStep {
    int iteration = 0;
    double energy = 0.0;
    double gradient_norm = 0.0; // max-norm over the free DOFs
    bool pattern_changed = false;
    int diagonal_shifts = 0;
    double feasible_step = 0.0; // max_feasible_step's bound on the step
    double step = 0.0;          // the accepted step length; 0 when none was
    int line_search_evaluations = 0;
};

struct NewtonResult {
    NewtonStatus status = NewtonStatus::MaxIterations;
    Vec x;
    double energy = 0.0;
    double gradient_norm = 0.0;
    int iterations = 0; // accepted steps
};

// Minimizes energy over the free DOFs of fixed, starting from x0. Fixed DOFs keep
// their x0 values exactly. solver factors the free block of the Hessian; it is
// analyzed at the start and whenever the energy's pattern changes.
//
// Each iteration: update the pattern, eval value, gradient and Hessian once,
// stop if converged, solve for a descent direction (shifting the diagonal when the
// system does not factor or the solution is not a descent direction), then search
// along it: start at max_feasible_step, shrink until the energy decreases or stays
// within floating-point noise of it. Evaluation exceptions propagate immediately;
// a std::domain_error trial point aborts the solve without retrying.
//
// Outcomes are statuses. Any infeasible state throws std::domain_error; a broken energy
// (a non-finite gradient) throws std::logic_error.
NewtonResult newton_solve(Energy& energy, const DofReduction& fixed, const ConstVecRef& x0,
                          const NewtonOptions& options, SparseSolver& solver,
                          const std::function<void(const NewtonStep&)>& on_iteration = {});

} // namespace litepgo
