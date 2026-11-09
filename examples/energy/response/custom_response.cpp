// Subclass fixed/dynamic Response and pass the residual object to Gauss-Newton.
// Build: cmake --build build/release --target custom_response
// Run: build/release/examples/energy/response/custom_response

#include "litepgo/energy/response/dynamic_pattern_response.hpp"
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include <iostream>
#include <stdexcept>

using namespace litepgo;

class LinearResidual final : public energy::FixedPatternResponse {
  public:
    LinearResidual() : FixedPatternResponse(pattern()) {}

  private:
    static SpMat pattern() {
        Eigen::Matrix<double, 3, 2> J;
        J << 1, 0, 0, 1, 1, 1;
        return J.sparseView();
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        if (J)
            J->coeffs().setOnes();
        return (Vec(3) << x[0] - 1, x[1] - 2, x.sum() - 3).finished();
    }
};

// Two outputs and two inputs; an off-diagonal derivative activates for x0 > 0.
class NonlinearResidual final : public energy::DynamicPatternResponse {
  public:
    NonlinearResidual() : DynamicPatternResponse(2, 2) {}

  private:
    SpMat jacobian_pattern_at(const ConstVecRef& x) const override {
        SpMat p = Eigen::Matrix2d::Identity().sparseView();
        if (x[0] > 0)
            p.insert(1, 0) = 1;
        p.makeCompressed();
        return p;
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        const double s = std::max(0., x[0]);
        if (J) {
            if (s > 0 && J->nonZeros() != 3)
                throw std::logic_error("Incorrect Jacobian structure");
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = it.row() == c ? 1. : 2 * s;
        }
        return (Vec(2) << x[0] - 1, x[1] + s * s - 2).finished();
    }
};

int main() {
    LinearResidual fixed;
    auto evaluation = fixed.evaluate(Eigen::Vector2d(1, 2));
    const auto J = evaluation->jacobian();
    const Vec r = evaluation->value();
    NonlinearResidual dynamic;
    const Vec active = Vec::Ones(2);
    dynamic.evaluate(active)->jacobian();
    dynamic.evaluate(-active)->jacobian();
    const SpMat buffer = dynamic.evaluate(2 * active)->jacobian();
    if (buffer.coeff(1, 0) != 4)
        throw std::runtime_error("Response Jacobian evaluation failed");
    solver::GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    const auto result = solver::gauss_newton_solve(dynamic, -active, options);
    if (result.status != solver::GaussNewtonStatus::Converged)
        throw std::runtime_error("Response fit failed");
    std::cout << "fixed Jacobian=" << J.rows() << "x" << J.cols() << ", dynamic solution=" << result.x.transpose()
              << ", Jacobian nnz=" << dynamic.evaluate(result.x)->jacobian().nonZeros() << '\n';
}
