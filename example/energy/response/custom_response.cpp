// Subclass fixed/dynamic Response and pass the residual object to Gauss-Newton.
// Build: cmake --build build/release --target custom_response
// Run: build/release/example/energy/response/custom_response

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
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override {
        if (J)
            J->coeffs().setOnes();
        return (Vec(3) << x[0] - 1, x[1] - 2, x.sum() - 3).finished();
    }
};

// Two outputs and two inputs; an off-diagonal derivative activates for x0 > 0.
class NonlinearResidual final : public energy::DynamicPatternResponse {
  public:
    NonlinearResidual() : DynamicPatternResponse(Eigen::Matrix2d::Identity().sparseView()) {}

  private:
    SpMat pattern_at(const ConstVecRef& x) const override {
        SpMat p = Eigen::Matrix2d::Identity().sparseView();
        if (x[0] > 0)
            p.insert(1, 0) = 1;
        p.makeCompressed();
        return p;
    }
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override {
        const double s = std::max(0., x[0]);
        if (J) {
            if (s > 0 && J->nonZeros() != 3)
                throw std::logic_error("Call update_pattern before Jacobian evaluation");
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = it.row() == c ? 1. : 2 * s;
        }
        return (Vec(2) << x[0] - 1, x[1] + s * s - 2).finished();
    }
};

int main() {
    LinearResidual fixed;
    SpMat J = fixed.jacobian_pattern();
    const Vec r = fixed.eval(Eigen::Vector2d(1, 2), &J);
    if (r.norm() != 0 || J.rows() != 3 || J.cols() != 2 || J.nonZeros() != 4)
        throw std::runtime_error("Rectangular response check failed");
    Eigen::Matrix<double, 3, 2> expected;
    expected << 1, 0, 0, 1, 1, 1;
    if ((Eigen::MatrixXd(J) - expected).norm() != 0)
        throw std::runtime_error("Rectangular Jacobian values failed");
    NonlinearResidual dynamic;
    const Vec active = Vec::Ones(2);
    if (dynamic.eval_value(active).norm() != 0 || !dynamic.update_pattern(active))
        throw std::runtime_error("Dynamic response check failed");
    J = dynamic.eval_jacobian(active);
    if (J.nonZeros() != 3 || (Eigen::MatrixXd(J) - (Eigen::Matrix2d() << 1, 0, 2, 1).finished()).norm() != 0)
        throw std::runtime_error("Pattern change check failed");
    dynamic.eval(Vec::Zero(2), &J);
    if ((Eigen::MatrixXd(J) - Eigen::Matrix2d::Identity()).norm() != 0 || J.nonZeros() != 3)
        throw std::runtime_error("Boundary Jacobian or stale values failed");
    if (!dynamic.update_pattern(-active))
        throw std::runtime_error("Pattern shrink failed");
    bool rejected = false;
    try {
        dynamic.eval_jacobian(active);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Missing pattern update was not rejected");
    solver::GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    const auto result = solver::gauss_newton_solve(dynamic, -active, options);
    if (result.status != solver::GaussNewtonStatus::Converged || (result.x - active).norm() > 1e-8)
        throw std::runtime_error("Response fit failed");
    std::cout << "fixed Jacobian=3x2, dynamic solution=" << result.x.transpose()
              << ", Jacobian nnz=" << dynamic.jacobian_pattern().nonZeros() << '\n';
}
