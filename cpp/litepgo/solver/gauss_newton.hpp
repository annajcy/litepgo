#pragma once

#include "litepgo/core/types.hpp"
#include <functional>
#include <limits>

namespace litepgo::solver {

// Returns a nonempty residual. If requested, fills its analytic Jacobian
// (residuals by parameters). Dimensions stay fixed throughout a solve.
// Non-finite outputs are errors. domain_error at a trial means infeasible;
// exceptions at the current point and all other exceptions propagate.
using ResidualFunction = std::function<Vec(const ConstVecRef&, Eigen::MatrixXd*)>;

struct GaussNewtonOptions {
    double gradient_tolerance = std::numeric_limits<double>::quiet_NaN();
    int max_iterations = 50;
    int max_trial_steps = 20;
    double initial_damping = 1e-3;
};
enum class GaussNewtonStatus { Converged, MaxIterations, StepTooSmall };
struct GaussNewtonResult {
    Vec x;
    double objective = 0;
    double gradient_norm = 0;
    int iterations = 0;
    int evaluations = 0;
    GaussNewtonStatus status = GaussNewtonStatus::MaxIterations;
};

// Minimizes 0.5*||r||^2 with dense damped Gauss-Newton. No numerical derivatives.
// Trials request only residuals. Convergence means ||J^T r||_inf <= tolerance,
// not a zero residual or a guarantee of a local minimum. Callbacks must be deterministic.
GaussNewtonResult gauss_newton_solve(const ResidualFunction& residual, const ConstVecRef& initial,
                                     const GaussNewtonOptions& options);

} // namespace litepgo::solver
