// Fit y(t) = amplitude * exp(-rate * t) with an analytic Jacobian.
// Build: cmake --build build/release --target gauss_newton_fit
// Run: build/release/example/solver/gauss_newton_fit

#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include <iostream>
#include <stdexcept>

using namespace litepgo;

class ExponentialResidual final : public energy::FixedPatternResponse {
  public:
    ExponentialResidual()
        : FixedPatternResponse(Eigen::MatrixXd::Ones(9, 2).sparseView()), t(Vec::LinSpaced(9, 0., 2.)),
          observations(2.5 * (-0.8 * t.array()).exp()) {}

  private:
    Vec eval_values(const ConstVecRef& parameters, SpMat* jacobian) const override {
        const double amplitude = parameters[0], rate = parameters[1];
        const Vec decay = (-rate * t.array()).exp();
        if (jacobian) {
            for (int i = 0; i < t.size(); ++i) {
                jacobian->coeffRef(i, 0) = decay[i];
                jacobian->coeffRef(i, 1) = -amplitude * t[i] * decay[i];
            }
        }
        return amplitude * decay - observations;
    }
    Vec t, observations;
};

int main() {
    ExponentialResidual residual;
    const Vec initial = Eigen::Vector2d(1., 0.3);
    const auto result = solver::gauss_newton_solve(residual, initial, {.gradient_tolerance = 1e-10});
    if (result.status != solver::GaussNewtonStatus::Converged ||
        (result.x - Eigen::Vector2d(2.5, 0.8)).lpNorm<Eigen::Infinity>() > 1e-8 || result.objective >= 1e-16)
        throw std::runtime_error("Exponential fit check failed");
    std::cout << "amplitude=" << result.x[0] << ", rate=" << result.x[1] << ", objective=" << result.objective
              << ", iterations=" << result.iterations << '\n';
}
