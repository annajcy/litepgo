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
    Quadratic() : FixedPatternEnergy(Eigen::Matrix2d::Identity().sparseView()) {}

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
    Coupled() : DynamicPatternEnergy(2) {}

  private:
    SpMat hessian_pattern_at(const ConstVecRef& x) const override {
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
                throw std::logic_error("Invalid prepared Hessian structure");
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
    const auto fixed_result = fixed.eval(x, true, energy::HessianKind::Exact);
    Coupled dynamic;
    const auto active = dynamic.eval(x, true, energy::HessianKind::Exact);
    const auto inactive = dynamic.eval_hessian(-x);
    std::cout << "fixed value=" << fixed_result.value << ", active value=" << active.value << ", Hessian nnz=2->"
              << active.hessian->nonZeros() << "->" << inactive.nonZeros() << '\n';
}
