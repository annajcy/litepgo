// Minimize 1/2 sum (x[i+1] - x[i])**2 with prescribed endpoints.
// Build: cmake --build build/release --target newton_chain
// Run: build/release/examples/solver/newton_chain

#include "litepgo/energy/scalar/quadratic_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/solver/newton.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace litepgo;
    constexpr int n = 6;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n - 1; ++i) {
        entries.emplace_back(i, i, -1.);
        entries.emplace_back(i, i + 1, 1.);
    }
    SpMat differences(n - 1, n);
    differences.setFromTriplets(entries.begin(), entries.end());
    // E(x) = 1/2 sum (x[i+1] - x[i])^2; the endpoints remove translation freedom.
    energy::QuadraticEnergy energy(differences.transpose() * differences);
    Vec initial = Vec::Zero(n);
    initial[0] = 1.;
    initial[n - 1] = 3.;
    // Fixed indices live in DofReduction; their values are taken from initial.
    dof::DofReduction fixed(n, {0, n - 1});
    linalg::AutoSparseSolver linear_solver;
    const auto result = solver::newton_solve(
        energy, fixed, initial, {.gradient_tolerance = 1e-10}, linear_solver, [](const solver::NewtonStep& step) {
            std::cout << "iteration=" << step.iteration << ", energy=" << step.energy
                      << ", free_gradient=" << step.gradient_norm << ", step=" << step.step << '\n';
        });
    if (result.status != solver::NewtonStatus::Converged || result.x[0] != initial[0] ||
        result.x[n - 1] != initial[n - 1] || !result.x.isApprox(Vec::LinSpaced(n, 1., 3.), 1e-10))
        throw std::runtime_error("Chain equilibrium check failed");
    std::cout << "equilibrium=" << result.x.transpose() << '\n';
}
