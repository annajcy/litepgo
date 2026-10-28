#include "litepgo/solver/newton.hpp"

#include "litepgo/linalg/reduced_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace litepgo {

namespace {

constexpr int max_diagonal_shifts = 10;

void check_arguments(const Energy& energy, const DofReduction& fixed, const ConstVecRef& x0,
                     const NewtonOptions& options) {
    if (fixed.num_dofs() != energy.num_dofs() || x0.size() != energy.num_dofs()) {
        throw std::invalid_argument("newton_solve: energy, fixed DOFs, and x0 must have the same DOF count");
    }
    if (!(options.gradient_tolerance >= 0.0) || !std::isfinite(options.gradient_tolerance)) {
        throw std::invalid_argument("newton_solve: gradient_tolerance must be set to a finite non-negative value");
    }
    if (options.max_iterations < 0 || options.max_line_search_steps < 1) {
        throw std::invalid_argument("newton_solve: max_iterations must be >= 0 and max_line_search_steps >= 1");
    }
    if (!(options.line_search_shrink > 0.0 && options.line_search_shrink < 1.0)) {
        throw std::invalid_argument("newton_solve: line_search_shrink must lie in (0, 1)");
    }
}

// Termination: the free gradient's max-norm is within the tolerance.
bool is_converged(const NewtonStep& step, const NewtonOptions& options) {
    return step.gradient_norm <= options.gradient_tolerance;
}

// Evaluates value, gradient, and Hessian at x in one assembly, and the free gradient.
double eval_state(const Energy& energy, const DofReduction& fixed, const Vec& x, HessianKind kind, Vec& gradient,
                  SpMat& hessian, Vec& gradient_free) {
    const double value = energy.eval(x, &gradient, &hessian, kind);
    if (!gradient.allFinite()) {
        throw std::logic_error("newton_solve: the energy returned a non-finite gradient");
    }
    gradient_free = fixed.reduce(gradient);
    return value;
}

// The linear system of a solve: the free block of the Hessian with its whole
// diagonal, and the sparse solver analyzed for its pattern.
class LinearSystem {
  public:
    LinearSystem(const DofReduction& fixed, SparseSolver& solver) : fixed_(fixed), solver_(solver) {}

    // Rebuilds the free block and the analysis for a new Hessian pattern.
    void rebuild(const SpMat& hessian) {
        matrix_.emplace(fixed_, hessian);
        if (!solver_.analyze(matrix_->matrix())) {
            throw std::runtime_error("newton_solve: the sparse solver could not analyze the Hessian pattern");
        }
    }

    // Damping: solves H_ff d = -g for a descent direction, shifting the diagonal
    // while the system does not factor or d is not a descent direction. Returns
    // the number of shifts used, or nullopt when even the largest one failed.
    std::optional<int> descent_direction(const SpMat& hessian, const Vec& gradient_free, Vec& direction_free) {
        matrix_->assign(hessian);
        const double diagonal_scale = std::max(1.0, matrix_->matrix().diagonal().cwiseAbs().maxCoeff());
        double shift = 1e-8 * diagonal_scale;
        for (int shifts = 0;; ++shifts) {
            if (solver_.factorize(matrix_->matrix())) {
                direction_free = solver_.solve(-gradient_free);
                if (gradient_free.dot(direction_free) < 0.0) {
                    return shifts;
                }
            }
            if (shifts == max_diagonal_shifts) {
                return std::nullopt;
            }
            matrix_->add_to_diagonal(shift);
            shift *= 10.0;
        }
    }

  private:
    const DofReduction& fixed_;
    SparseSolver& solver_;
    std::optional<ReducedMatrix> matrix_;
};

// Accepts a trial energy that decreases, or that stays within floating-point noise
// of the current one: near convergence, true decreases fall below what the energy
// resolves. The direction is known to be a descent direction.
bool accepts(double energy, double trial_energy) {
    const double noise = 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(energy));
    return trial_energy <= energy + noise;
}

// Line search along a descent direction: starts at max_feasible_step and shrinks
// until the finite trial energy is accepted. Evaluation exceptions propagate;
// a step that no longer moves x ends the search. Leaves the
// accepted point in trial and returns its step length, or nullopt.
std::optional<double> line_search(const Energy& energy, const Vec& x, const Vec& direction,
                                  const NewtonOptions& options, Vec& trial, NewtonStep& step) {
    double alpha = energy.max_feasible_step(x, direction);
    step.feasible_step = alpha;
    for (int j = 0; j < options.max_line_search_steps; ++j) {
        trial = x + alpha * direction;
        if (trial == x) {
            return std::nullopt; // accepting a step that does not move x would make no progress
        }
        ++step.line_search_evaluations;
        if (accepts(step.energy, energy.eval_value(trial))) {
            return alpha;
        }
        alpha *= options.line_search_shrink;
    }
    return std::nullopt;
}

} // namespace

NewtonResult newton_solve(Energy& energy, const DofReduction& fixed, const ConstVecRef& x0,
                          const NewtonOptions& options, SparseSolver& solver,
                          const std::function<void(const NewtonStep&)>& on_iteration) {
    check_arguments(energy, fixed, x0, options);
    NewtonResult result;
    result.x = x0;
    if (fixed.num_free() == 0) {
        result.status = NewtonStatus::Converged;
        result.energy = energy.eval_value(result.x);
        return result;
    }

    const auto report = [&](const NewtonStep& step) {
        if (on_iteration) {
            on_iteration(step);
        }
    };
    const auto finish = [&](NewtonStatus status, const NewtonStep& step) {
        report(step);
        result.status = status;
        result.energy = step.energy;
        result.gradient_norm = step.gradient_norm;
        result.iterations = step.iteration;
        return result;
    };

    Vec& x = result.x;
    Vec gradient;
    Vec gradient_free;
    Vec direction_free;
    Vec direction = Vec::Zero(x.size()); // fixed entries stay zero
    Vec trial(x.size());
    SpMat hessian;
    LinearSystem system(fixed, solver);

    for (int iteration = 0;; ++iteration) {
        NewtonStep step;
        step.iteration = iteration;

        step.pattern_changed = energy.update_pattern(x);
        if (iteration == 0 || step.pattern_changed) {
            hessian = energy.hessian_pattern();
            system.rebuild(hessian);
        }
        step.energy = eval_state(energy, fixed, x, options.hessian, gradient, hessian, gradient_free);
        step.gradient_norm = gradient_free.cwiseAbs().maxCoeff();

        if (is_converged(step, options)) {
            return finish(NewtonStatus::Converged, step);
        }
        if (iteration == options.max_iterations) {
            return finish(NewtonStatus::MaxIterations, step);
        }

        const std::optional<int> shifts = system.descent_direction(hessian, gradient_free, direction_free);
        step.diagonal_shifts = shifts.value_or(max_diagonal_shifts);
        if (!shifts) {
            return finish(NewtonStatus::NoDescentDirection, step);
        }
        fixed.expand(direction_free, direction);

        const std::optional<double> alpha = line_search(energy, x, direction, options, trial, step);
        if (!alpha) {
            return finish(NewtonStatus::StepTooSmall, step);
        }
        step.step = *alpha;
        report(step);
        x = trial;
    }
}

} // namespace litepgo
