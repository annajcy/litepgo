#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"
#include "litepgo/energy/response/response.hpp"
#include <limits>

namespace litepgo::solver {

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

// Minimizes 0.5*||r||^2 with sparse normal-equation damped Gauss-Newton. No numerical derivatives.
// Solves (J^T J + damping I) p = -J^T r; scale inputs and residuals appropriately.
// Symbolic analysis is reused while the normal-matrix pattern is unchanged.
// The three-argument overload uses AutoSparseSolver.
// Trials request only residuals. Convergence means ||J^T r||_inf <= tolerance,
// not a zero residual or a guarantee of a local minimum. Responses must be deterministic. The value is interpreted as a
// residual targeting zero.
GaussNewtonResult gauss_newton_solve(energy::Response& residual, const ConstVecRef& initial,
                                     const GaussNewtonOptions& options);

GaussNewtonResult gauss_newton_solve(energy::Response& residual, const ConstVecRef& initial,
                                     const GaussNewtonOptions& options, linalg::SparseSolver& linear_solver);

} // namespace litepgo::solver
