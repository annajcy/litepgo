// Subclass fixed/dynamic Energy; request values, gradients and sparse Hessians.
// Build: cmake --build build/release --target custom_energy
// Run: build/release/example/energy/scalar/custom_energy

#include "litepgo/energy/scalar/dynamic_pattern_energy.hpp"
#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"
#include <iostream>
#include <stdexcept>

using namespace litepgo;

// E(x) = 0.5 ||x - (1,2)||^2.
class Quadratic final : public energy::FixedPatternEnergy {
  public:
    Quadratic() : FixedPatternEnergy(2, Eigen::Matrix2d::Identity().sparseView()) {}

  private:
    double eval_values(const ConstVecRef& x, Vec* g, SpMat* H, energy::HessianKind) const override {
        const Vec r = x - Eigen::Vector2d(1, 2);
        if (g)
            *g = r;
        if (H) {
            H->coeffRef(0, 0) = 1;
            H->coeffRef(1, 1) = 1;
        }
        return 0.5 * r.squaredNorm();
    }
};

// E(x) = 0.5 ||x||^2 + 0.25 max(x0+x1,0)^4; C2 across the activation boundary.
class Coupled final : public energy::DynamicPatternEnergy {
  public:
    Coupled() : DynamicPatternEnergy(2, Eigen::Matrix2d::Identity().sparseView()) {}

  private:
    SpMat pattern_at(const ConstVecRef& x) const override {
        if (x.sum() > 0)
            return Eigen::Matrix2d::Ones().sparseView();
        return Eigen::Matrix2d::Identity().sparseView();
    }
    double eval_values(const ConstVecRef& x, Vec* g, SpMat* H, energy::HessianKind) const override {
        const double s = std::max(0., x.sum());
        if (g)
            *g = x.array() + s * s * s;
        if (H) {
            if (s > 0 && H->nonZeros() != 4)
                throw std::logic_error("Call update_pattern before Hessian evaluation");
            for (int c = 0; c < H->outerSize(); ++c)
                for (SpMat::InnerIterator it(*H, c); it; ++it)
                    it.valueRef() = (it.row() == c ? 1. : 0.) + 3 * s * s;
        }
        // Both Hessians are PSD, so Exact and Projected are identical here.
        return 0.5 * x.squaredNorm() + 0.25 * s * s * s * s;
    }
};

int main() {
    Quadratic fixed;
    const Vec x = Eigen::Vector2d(1, 1);
    Vec g;
    SpMat H = fixed.hessian_pattern();
    const double value = fixed.eval(x, &g, &H);
    if (value != 0.5 || (g - Eigen::Vector2d(0, -1)).norm() != 0 ||
        (Eigen::MatrixXd(H) - Eigen::Matrix2d::Identity()).norm() != 0)
        throw std::runtime_error("Fixed energy check failed");
    Coupled dynamic;
    // Value/gradient requests do not need a pattern update.
    if (dynamic.eval_value(x) != 5.)
        throw std::runtime_error("Dynamic value check failed");
    if (!dynamic.update_pattern(x))
        throw std::runtime_error("Expected pattern growth");
    H = dynamic.hessian_pattern(); // Refresh the buffer after a structural change.
    dynamic.eval(x, &g, &H);
    if ((g - Vec::Constant(2, 9)).norm() != 0 ||
        (Eigen::MatrixXd(H) - (Eigen::Matrix2d() << 13, 12, 12, 13).finished()).norm() != 0 || H.nonZeros() != 4)
        throw std::runtime_error("Dynamic derivative check failed");
    // A larger cached pattern must not retain active values at an inactive state.
    dynamic.eval(-x, nullptr, &H);
    if ((Eigen::MatrixXd(H) - Eigen::Matrix2d::Identity()).norm() != 0 || H.nonZeros() != 4)
        throw std::runtime_error("Stale Hessian values");
    const Vec boundary = Eigen::Vector2d(1, -1);
    dynamic.eval(boundary, &g, &H, energy::HessianKind::Projected);
    if ((g - boundary).norm() != 0 || (Eigen::MatrixXd(H) - Eigen::Matrix2d::Identity()).norm() != 0)
        throw std::runtime_error("Boundary derivative check failed");
    if (dynamic.update_pattern(x) || !dynamic.update_pattern(-x))
        throw std::runtime_error("Pattern update check failed");
    H = dynamic.eval_hessian(-x);
    if ((Eigen::MatrixXd(H) - Eigen::Matrix2d::Identity()).norm() != 0)
        throw std::runtime_error("Inactive Hessian check failed");
    bool rejected = false;
    try {
        dynamic.eval_hessian(x);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Missing pattern update was not rejected");
    std::cout << "fixed value=" << value << ", active value=5, Hessian nnz=2->4->" << H.nonZeros() << '\n';
}
