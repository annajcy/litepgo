#include "litepgo/solver/gauss_newton.hpp"
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace litepgo::solver {
GaussNewtonResult gauss_newton_solve(const ResidualFunction& residual, const ConstVecRef& initial,
                                     const GaussNewtonOptions& options) {
    if (!residual || initial.size() == 0 || !initial.allFinite() || !std::isfinite(options.gradient_tolerance) ||
        options.gradient_tolerance <= 0 || options.max_iterations < 0 || options.max_trial_steps <= 0 ||
        !std::isfinite(options.initial_damping) || options.initial_damping <= 0)
        throw std::invalid_argument("Gauss-Newton: invalid input or options");
    GaussNewtonResult result;
    result.x = initial;
    Eigen::Index count = -1;
    auto evaluate = [&](const Vec& x, Eigen::MatrixXd* J) {
        ++result.evaluations;
        Vec r = residual(x, J);
        if (count < 0)
            count = r.size();
        if (r.size() == 0 || r.size() != count || !r.allFinite() || !std::isfinite(r.squaredNorm()) ||
            (J && (J->rows() != count || J->cols() != initial.size() || !J->allFinite())))
            throw std::logic_error("Gauss-Newton: invalid residual or Jacobian");
        return r;
    };
    double damping = options.initial_damping;
    for (;;) {
        Eigen::MatrixXd J;
        const Vec r = evaluate(result.x, &J);
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
        bool accepted = false;
        for (int trial = 0; trial < options.max_trial_steps && std::isfinite(damping); ++trial) {
            // Augmented QR solves the damped system without squaring J's condition number.
            Eigen::MatrixXd A(count + initial.size(), initial.size());
            A.topRows(count) = J;
            A.bottomRows(initial.size()) =
                std::sqrt(damping) * Eigen::MatrixXd::Identity(initial.size(), initial.size());
            Vec rhs = Vec::Zero(A.rows());
            rhs.head(count) = -r;
            const Vec step = A.colPivHouseholderQr().solve(rhs);
            const Vec next_x = result.x + step;
            if (step.allFinite() && next_x.allFinite() && (next_x.array() != result.x.array()).any()) {
                Vec next;
                bool feasible = true;
                try {
                    next = evaluate(next_x, nullptr);
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
