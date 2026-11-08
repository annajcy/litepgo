#include <memory>
// Weighted energies in reduced coordinates x = T z + offset.
// Build: cmake --build build/release --target energy_composition
// Run: build/release/examples/energy/scalar/energy_composition

#include "litepgo/energy/scalar/affine_mapped_energy.hpp"
#include "litepgo/energy/scalar/quadratic_energy.hpp"
#include "litepgo/energy/scalar/vertex_pulling_energy.hpp"
#include "litepgo/energy/scalar/weighted_sum_energy.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace litepgo;
    const Vec target = Eigen::Vector3d(1, 2, 3);
    SpMat identity(3, 3);
    identity.setIdentity();
    auto regularizer = std::make_shared<energy::QuadraticEnergy>(identity);
    auto pulling =
        std::make_shared<energy::VertexPullingEnergy>(3, std::vector<std::array<int, 3>>{{0, 1, 2}}, target, 1.);
    // All terms share the same DOFs; weights scale their values and derivatives.
    auto total = std::make_shared<energy::WeightedSumEnergy>(
        std::vector<energy::WeightedEnergy>{{regularizer, 0.5}, {pulling, 2.}});
    Eigen::Matrix<double, 3, 2> T;
    T << 1, 0, 0, 1, 1, 1;
    const Vec offset = Eigen::Vector3d(0, 0, 1);
    energy::AffineMappedEnergy mapped(total, T.sparseView(), offset);
    const Vec z = Eigen::Vector2d(0.2, -0.3);
    // Joint evaluation prepares structure and owns all requested outputs.
    auto result = mapped.eval(z, true, energy::HessianKind::Exact);
    const double value = result.value;
    const Vec& gradient = *result.gradient;
    const SpMat& hessian = *result.hessian;
    const Vec x = T * z + offset;
    if (std::abs(value - (0.25 * x.squaredNorm() + (x - target).squaredNorm())) > 1e-12 ||
        !gradient.isApprox(T.transpose() * (2.5 * x - 2. * target), 1e-12) ||
        !Eigen::MatrixXd(hessian).isApprox(2.5 * T.transpose() * T, 1e-12))
        throw std::runtime_error("Energy composition check failed");
    std::cout << "energy=" << value << ", gradient=" << gradient.transpose() << ", Hessian=\n"
              << Eigen::MatrixXd(hessian) << '\n';
}
