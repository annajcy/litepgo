#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#include <functional>
#include <limits>
#include <optional>

namespace litepgo::solver {

// Passed by value so language callbacks may retain an immutable snapshot.
struct NewtonConvergenceState {
    int iteration = 0;
    double energy = 0.0;
    double gradient_norm = 0.0;
    bool project_hessian = true;
    std::optional<double> newton_correction_norm;
    std::optional<int> diagonal_shifts;
    std::optional<double> linear_backward_error;
};
using ConvergenceCheck = std::function<bool(NewtonConvergenceState)>;

class AbsoluteConvergence {
  public:
    explicit AbsoluteConvergence(double tolerance);
    double tolerance() const noexcept {
        return tolerance_;
    }
    bool operator()(const NewtonConvergenceState& state) const {
        return state.gradient_norm <= tolerance_;
    }

  private:
    double tolerance_;
};

// Absolute residual OR relative residual with a reliable, small full Newton correction.
class CombinedConvergence {
  public:
    CombinedConvergence(double absolute_tolerance, double relative_tolerance, double force_scale,
                        double correction_tolerance, double coordinate_scale, double backward_error_tolerance);
    bool operator()(const NewtonConvergenceState& state) const;
    const double absolute_tolerance, relative_tolerance, force_scale;
    const double correction_tolerance, coordinate_scale, backward_error_tolerance;
};

struct NewtonOptions {
    int max_iterations = 50;
    bool project_hessian = true;
    int max_line_search_steps = 30;
    double line_search_shrink = 0.5;
    // Bounds max_i |x_next[i] - x[i]| per accepted iteration in solver coordinates.
    // Scales the whole direction before CCD; infinity disables the bound.
    double max_coordinate_step = std::numeric_limits<double>::infinity();
    // Opt-in diagnostic, not convergence; only unshifted exact Newton directions qualify.
    bool detect_precision_stagnation = false;
    // Positive characteristic scale in solver coordinates (including near-zero DOFs).
    double precision_coordinate_scale = 1.0;
};

enum class NewtonStatus {
    // The caller-provided convergence check accepted the current state.
    Converged,
    MaxIterations,
    // The line search found no acceptable point within max_line_search_steps.
    StepTooSmall,
    // Even with the largest diagonal shift, the system would not factor or its
    // solution was not a descent direction.
    NoDescentDirection,
    // Requested tolerance was NOT met; reliable full Newton steps reached roundoff scale.
    PrecisionStagnation,
};

// What one iteration did, as reported to the callback.
struct NewtonStep {
    int iteration = 0;
    double energy = 0.0;
    double gradient_norm = 0.0; // max-norm over the free DOFs
    // True on the first Hessian assembly and whenever its structure changes.
    bool pattern_changed = false;
    int diagonal_shifts = 0;
    double feasible_step = 0.0; // combined coordinate and feasibility bound on the original direction
    double step = 0.0;          // the accepted step length; 0 when none was
    int line_search_evaluations = 0;
    double scaled_direction_norm = std::numeric_limits<double>::quiet_NaN();
    double linear_backward_error = std::numeric_limits<double>::quiet_NaN();
    int precision_stagnation_steps = 0;
};

struct NewtonResult {
    NewtonStatus status = NewtonStatus::MaxIterations;
    Vec x;
    double energy = 0.0;
    double gradient_norm = 0.0;
    int iterations = 0; // accepted steps
    double scaled_direction_norm = std::numeric_limits<double>::quiet_NaN();
    double linear_backward_error = std::numeric_limits<double>::quiet_NaN();
    int precision_stagnation_steps = 0;
};

// Minimizes energy over the free DOFs of fixed, starting from x0. Fixed DOFs keep
// their x0 values exactly. solver factors the free block of the Hessian; it is
// analyzed at the start and whenever the energy's pattern changes.
//
// Checks initial feasibility with a zero-direction max_feasible_step query,
// including when all DOFs are fixed. Check convergence after value and gradient;
// if rejected, assemble the Hessian, solve for a descent direction (shifting when
// needed), and check convergence again with direction diagnostics. Then search
// along it: cap coordinate movement before querying max_feasible_step, then shrink
// until the energy decreases or stays within floating-point noise of it. Evaluation exceptions propagate immediately;
// a std::domain_error trial point aborts the solve without retrying.
//
// Outcomes are statuses. Any infeasible state throws std::domain_error; a broken energy
// (a non-finite gradient) throws std::logic_error.
NewtonResult newton_solve(energy::Energy& energy, const dof::DofReduction& fixed, const ConstVecRef& x0,
                          const NewtonOptions& options, linalg::SparseSolver& solver,
                          const ConvergenceCheck& convergence,
                          const std::function<void(const NewtonStep&)>& on_iteration = {},
                          std::shared_ptr<energy::EnergyEvaluation>* converged_evaluation = nullptr);

} // namespace litepgo::solver
