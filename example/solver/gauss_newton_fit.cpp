#include "litepgo/solver/gauss_newton.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace litepgo;
    const Vec t = Vec::LinSpaced(9, 0., 2.);
    const Vec observations = 2.5 * (-0.8 * t.array()).exp();
    const auto residual = [&](const ConstVecRef& parameters, Eigen::MatrixXd* jacobian) -> Vec {
        const double amplitude = parameters[0], rate = parameters[1];
        const Vec decay = (-rate * t.array()).exp();
        // Trial points request residuals only; skip the Jacobian in that case.
        if (jacobian) {
            jacobian->resize(t.size(), 2);
            jacobian->col(0) = decay;
            jacobian->col(1) = -amplitude * t.array() * decay.array();
        }
        return amplitude * decay - observations;
    };
    const Vec initial = Eigen::Vector2d(1., 0.3);
    const auto result = solver::gauss_newton_solve(residual, initial, {.gradient_tolerance = 1e-10});
    if (result.status != solver::GaussNewtonStatus::Converged ||
        (result.x - Eigen::Vector2d(2.5, 0.8)).lpNorm<Eigen::Infinity>() > 1e-8 || result.objective >= 1e-16)
        throw std::runtime_error("Exponential fit check failed");
    std::cout << "amplitude=" << result.x[0] << ", rate=" << result.x[1] << ", objective=" << result.objective
              << ", iterations=" << result.iterations << '\n';
}
