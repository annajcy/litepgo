// Subclass fixed/dynamic Energy; request values, gradients and sparse Hessians.
// Build: cmake --build build/release --target custom_energy
// Run: build/release/examples/energy/scalar/custom_energy

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
    Coupled dynamic;
    const double active_value = dynamic.eval_value(x);
    dynamic.update_pattern(x);
    H = dynamic.hessian_pattern(); // Refresh the buffer after a structural change.
    dynamic.eval(x, &g, &H);
    const auto active_nnz = H.nonZeros();
    dynamic.update_pattern(-x);
    H = dynamic.eval_hessian(-x);
    std::cout << "fixed value=" << value << ", active value=" << active_value << ", Hessian nnz=2->" << active_nnz
              << "->" << H.nonZeros() << '\n';
}
