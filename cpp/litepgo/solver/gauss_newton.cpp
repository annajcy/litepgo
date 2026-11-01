#include "litepgo/solver/gauss_newton.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litepgo::solver {
GaussNewtonResult gauss_newton_solve(response::Response& residual, const ConstVecRef& initial,
                                     const GaussNewtonOptions& options) {
    linalg::AutoSparseSolver linear_solver;
    return gauss_newton_solve(residual, initial, options, linear_solver);
}

GaussNewtonResult gauss_newton_solve(response::Response& residual, const ConstVecRef& initial,
                                     const GaussNewtonOptions& options, linalg::SparseSolver& linear_solver) {
    if (initial.size() != residual.num_inputs() || residual.num_outputs() == 0 || initial.size() == 0 ||
        !initial.allFinite() || !std::isfinite(options.gradient_tolerance) || options.gradient_tolerance <= 0 ||
        options.max_iterations < 0 || options.max_trial_steps <= 0 || !std::isfinite(options.initial_damping) ||
        options.initial_damping <= 0)
        throw std::invalid_argument("Gauss-Newton: invalid input or options");
    GaussNewtonResult result;
    result.x = initial;
    SpMat J, analyzed_pattern;
    auto evaluate = [&](const Vec& x, bool derivatives) {
        ++result.evaluations;
        if (derivatives) {
            if (residual.update_pattern(x) || J.rows() == 0) {
                J = residual.jacobian_pattern();
            }
        }
        Vec r = residual.eval(x, derivatives ? &J : nullptr);
        if (!std::isfinite(r.squaredNorm()))
            throw std::logic_error("Gauss-Newton: residual norm overflow");
        return r;
    };
    double damping = options.initial_damping;
    for (;;) {
        const Vec r = evaluate(result.x, true);
        const Vec gradient = J.transpose() * r;
        if (!gradient.allFinite())
            throw std::runtime_error("Gauss-Newton: gradient overflow");
        result.objective = 0.5 * r.squaredNorm();
        result.gradient_norm = gradient.lpNorm<Eigen::Infinity>();
        if (result.gradient_norm <= options.gradient_tolerance) {
            result.status = GaussNewtonStatus::Converged;
            return result;
        }
        if (result.iterations == options.max_iterations)
            return result;
        SpMat normal = J.transpose() * J;
        // Include every diagonal even for parameters with no current influence.
        for (int i = 0; i < normal.cols(); ++i)
            normal.coeffRef(i, i) += 0.0;
        normal.makeCompressed();
        if (!normal.coeffs().allFinite())
            throw std::runtime_error("Gauss-Newton: normal matrix overflow");
        if (!linalg::same_pattern(normal, analyzed_pattern)) {
            if (!linear_solver.analyze(normal))
                throw std::runtime_error("Gauss-Newton: symbolic analysis failed");
            analyzed_pattern = linalg::pattern_of(normal);
        }
        const Vec rhs = -gradient;
        bool accepted = false;
        for (int trial = 0; trial < options.max_trial_steps && std::isfinite(damping); ++trial) {
            SpMat system = normal;
            for (int i = 0; i < system.cols(); ++i)
                system.coeffRef(i, i) += damping;
            if (!system.coeffs().allFinite())
                break;
            if (!linear_solver.factorize(system)) {
                damping *= 10;
                continue;
            }
            const Vec step = linear_solver.solve(rhs);
            const Vec next_x = result.x + step;
            if (step.allFinite() && next_x.allFinite() && (next_x.array() != result.x.array()).any()) {
                Vec next;
                bool feasible = true;
                try {
                    next = evaluate(next_x, false);
                } catch (const std::domain_error&) {
                    feasible = false;
                }
                if (feasible && 0.5 * next.squaredNorm() < result.objective) {
                    result.x = next_x;
                    ++result.iterations;
                    damping = std::max(std::numeric_limits<double>::min(), damping / 3);
                    accepted = true;
                    break;
                }
            }
            damping *= 10;
        }
        if (!accepted) {
            result.status = GaussNewtonStatus::StepTooSmall;
            return result;
        }
    }
}
} // namespace litepgo::solver
