#include "litepgo/solver/newton.hpp"

#include "litepgo/dof/reduced_matrix.hpp"
#include "litepgo/energy/scalar/detail/evaluation_access.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace litepgo::solver {

AbsoluteConvergence::AbsoluteConvergence(double tolerance) : tolerance_(tolerance) {
    if (!(tolerance >= 0.0) || !std::isfinite(tolerance))
        throw std::invalid_argument("AbsoluteConvergence: tolerance must be finite and non-negative");
}

namespace {

constexpr int max_diagonal_shifts = 10;

void check_arguments(const energy::Energy& energy, const dof::DofReduction& fixed, const ConstVecRef& x0,
                     const NewtonOptions& options) {
    if (fixed.num_dofs() != energy.num_dofs() || x0.size() != energy.num_dofs()) {
        throw std::invalid_argument("newton_solve: energy, fixed DOFs, and x0 must have the same DOF count");
    }
    if (options.max_iterations < 0 || options.max_line_search_steps < 1) {
        throw std::invalid_argument("newton_solve: max_iterations must be >= 0 and max_line_search_steps >= 1");
    }
    if (!(options.line_search_shrink > 0.0 && options.line_search_shrink < 1.0)) {
        throw std::invalid_argument("newton_solve: line_search_shrink must lie in (0, 1)");
    }
    if (!(options.precision_coordinate_scale > 0.0) || !std::isfinite(options.precision_coordinate_scale)) {
        throw std::invalid_argument("newton_solve: precision_coordinate_scale must be finite and positive");
    }
    if (!(options.max_coordinate_step > 0.0)) {
        throw std::invalid_argument("newton_solve: max_coordinate_step must be positive or infinity");
    }
}

// The linear system of a solve: the free block of the Hessian with its whole
// diagonal, and the sparse solver analyzed for its pattern.
class LinearSystem {
  public:
    LinearSystem(const dof::DofReduction& fixed, linalg::SparseSolver& solver) : fixed_(fixed), solver_(solver) {}

    bool has_pattern(const SpMat& hessian) const {
        return matrix_ && matrix_->has_pattern(hessian);
    }

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

    double backward_error(const Vec& gradient, const Vec& direction) const {
        const auto& H = matrix_->matrix();
        const double norm = (H.cwiseAbs() * Vec::Ones(H.cols())).maxCoeff();
        const double scale = norm * direction.cwiseAbs().maxCoeff() + gradient.cwiseAbs().maxCoeff();
        if (!(scale > 0.0) || !std::isfinite(scale))
            return std::numeric_limits<double>::infinity();
        return (H * direction + gradient).cwiseAbs().maxCoeff() / scale;
    }

  private:
    const dof::DofReduction& fixed_;
    linalg::SparseSolver& solver_;
    std::optional<dof::ReducedMatrix> matrix_;
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
std::optional<double> line_search(const energy::Energy& energy, const Vec& x, const Vec& direction,
                                  const NewtonOptions& options, Vec& trial, NewtonStep& step,
                                  const std::shared_ptr<energy::EnergyEvaluation>& current,
                                  std::shared_ptr<energy::EnergyEvaluation>& accepted) {
    const double scale = std::min(1.0, options.max_coordinate_step / direction.cwiseAbs().maxCoeff());
    double alpha = scale * current->max_feasible_step(scale * direction);
    step.feasible_step = alpha;
    for (int j = 0; j < options.max_line_search_steps; ++j) {
        trial = x + alpha * direction;
        if (trial == x) {
            return std::nullopt; // accepting a step that does not move x would make no progress
        }
        ++step.line_search_evaluations;
        auto evaluation = energy.evaluate(trial);
        if (accepts(step.energy, evaluation->value())) {
            accepted = std::move(evaluation);
            return alpha;
        }
        alpha *= options.line_search_shrink;
    }
    return std::nullopt;
}

} // namespace

NewtonResult newton_solve(energy::Energy& energy, const dof::DofReduction& fixed, const ConstVecRef& x0,
                          const NewtonOptions& options, linalg::SparseSolver& solver,
                          const ConvergenceCheck& convergence,
                          const std::function<void(const NewtonStep&)>& on_iteration,
                          std::shared_ptr<energy::EnergyEvaluation>* converged_evaluation) {
    if (!convergence)
        throw std::invalid_argument("newton_solve: convergence must be provided");
    check_arguments(energy, fixed, x0, options);
    NewtonResult result;
    result.x = x0;
    auto evaluation = energy.evaluate(result.x);
    evaluation->max_feasible_step(Vec::Zero(result.x.size()));
    if (fixed.num_free() == 0) {
        result.energy = evaluation->value();
        const NewtonConvergenceState state{0, result.energy, 0.0, options.project_hessian};
        result.status = convergence(state) ? NewtonStatus::Converged : NewtonStatus::NoDescentDirection;
        if (converged_evaluation && result.status == NewtonStatus::Converged)
            *converged_evaluation = evaluation;
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
        result.scaled_direction_norm = step.scaled_direction_norm;
        result.linear_backward_error = step.linear_backward_error;
        result.precision_stagnation_steps = step.precision_stagnation_steps;
        return result;
    };

    Vec& x = result.x;
    Vec gradient_free;
    Vec direction_free;
    Vec direction = Vec::Zero(x.size()); // fixed entries stay zero
    Vec trial(x.size());
    LinearSystem system(fixed, solver);
    int tiny_steps = 0;
    double best_tiny_residual = std::numeric_limits<double>::infinity();
    double previous_energy = std::numeric_limits<double>::quiet_NaN();

    for (int iteration = 0;; ++iteration) {
        NewtonStep step;
        step.iteration = iteration;

        gradient_free = fixed.reduce(energy::detail::EvaluationAccess::gradient(*evaluation));
        step.energy = evaluation->value();
        step.gradient_norm = gradient_free.cwiseAbs().maxCoeff();

        NewtonConvergenceState state{iteration, step.energy, step.gradient_norm, options.project_hessian};
        if (convergence(state)) {
            if (converged_evaluation)
                *converged_evaluation = evaluation;
            return finish(NewtonStatus::Converged, step);
        }
        if (iteration == options.max_iterations) {
            return finish(NewtonStatus::MaxIterations, step);
        }

        if (step.gradient_norm == 0.0) {
            return finish(NewtonStatus::NoDescentDirection, step);
        }
        const SpMat& hessian = options.project_hessian
                                   ? energy::detail::EvaluationAccess::projected_hessian(*evaluation)
                                   : energy::detail::EvaluationAccess::hessian(*evaluation);
        step.pattern_changed = !system.has_pattern(hessian);
        if (step.pattern_changed)
            system.rebuild(hessian);
        const std::optional<int> shifts = system.descent_direction(hessian, gradient_free, direction_free);
        step.diagonal_shifts = shifts.value_or(max_diagonal_shifts);
        if (!shifts) {
            return finish(NewtonStatus::NoDescentDirection, step);
        }
        fixed.expand(direction_free, direction);

        state.newton_correction_norm = direction_free.cwiseAbs().maxCoeff();
        state.diagonal_shifts = *shifts;
        state.linear_backward_error = system.backward_error(gradient_free, direction_free);
        step.linear_backward_error = *state.linear_backward_error;
        if (convergence(state)) {
            if (converged_evaluation)
                *converged_evaluation = evaluation;
            return finish(NewtonStatus::Converged, step);
        }
        bool reliable_tiny_direction = false;
        if (options.detect_precision_stagnation && !options.project_hessian && *shifts == 0) {
            step.scaled_direction_norm =
                (direction.array().abs() / x.array().abs().max(options.precision_coordinate_scale)).maxCoeff();
            reliable_tiny_direction = step.scaled_direction_norm <= 128 * std::numeric_limits<double>::epsilon() &&
                                      step.linear_backward_error <= 1e-12;
        }

        std::shared_ptr<energy::EnergyEvaluation> accepted;
        const std::optional<double> alpha =
            line_search(energy, x, direction, options, trial, step, evaluation, accepted);
        if (!alpha) {
            // A full feasible correction cannot move any coordinate; shrinking must not trigger this case.
            if (reliable_tiny_direction && step.feasible_step == 1.0 && Vec(x + direction) == x) {
                step.precision_stagnation_steps = tiny_steps + 1;
                return finish(NewtonStatus::PrecisionStagnation, step);
            }
            return finish(NewtonStatus::StepTooSmall, step);
        }
        const double energy_noise = 16 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(step.energy));
        if (reliable_tiny_direction && *alpha == 1.0 && std::abs(step.energy - previous_energy) <= energy_noise) {
            // Any new best residual restarts the window; steadily improving tiny steps must continue.
            if (step.gradient_norm < best_tiny_residual) {
                tiny_steps = 0;
                best_tiny_residual = step.gradient_norm;
            }
            step.precision_stagnation_steps = ++tiny_steps;
            if (tiny_steps >= 3)
                return finish(NewtonStatus::PrecisionStagnation, step);
        } else {
            tiny_steps = 0;
            best_tiny_residual = std::numeric_limits<double>::infinity();
        }
        previous_energy = step.energy;
        step.step = *alpha;
        report(step);
        x = trial;
        evaluation = std::move(accepted);
    }
}

} // namespace litepgo::solver
