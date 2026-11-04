// Batch callback assembly for a spring chain with fixed endpoints.
// Build: cmake --build build/release --target custom_mesh_energy
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/solver/newton.hpp"
#include <iostream>
#include <stdexcept>

using namespace litepgo;
int main() {
    const dof::StencilDofMap mapping(4, {{0, 1}, {1, 2}, {2, 3}}, 1);
    const Eigen::Vector3d stiffness(2, 3, 4), lengths(1, 2, 1);
    energy::CustomMeshEnergy potential(
        mapping,
        [stiffness, lengths](std::span<const int> ids, const energy::BatchMatrix& x, energy::BatchMatrix* g,
                             energy::BatchMatrix* H) {
            Vec values(ids.size());
            if (g)
                g->resize(x.rows(), 2);
            if (H)
                H->resize(2 * x.rows(), 2);
            for (Eigen::Index b = 0; b < x.rows(); ++b) {
                // ids select original edge data, independently of batch order.
                const double k = stiffness[ids[b]], stretch = x(b, 1) - x(b, 0) - lengths[ids[b]];
                values[b] = .5 * k * stretch * stretch;
                if (g)
                    g->row(b) << -k * stretch, k * stretch;
                if (H)
                    H->middleRows(2 * b, 2) << k, -k, -k, k;
            }
            return values;
        },
        2);
    // The polynomial has no restricted domain, so the default feasible step is one.
    Vec initial(4);
    initial << 0, 0, 0, 7;
    Vec g;
    SpMat H = potential.hessian_pattern();
    const double value = potential.eval(initial, &g, &H, energy::HessianKind::Exact);
    Eigen::Matrix<double, 3, 4> D;
    D << -1, 1, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1;
    const Eigen::Vector3d stretch = D * initial - lengths;
    if (std::abs(value - .5 * stiffness.dot(stretch.cwiseProduct(stretch))) > 1e-12 ||
        (g - D.transpose() * stiffness.cwiseProduct(stretch)).norm() > 1e-12 ||
        (Eigen::MatrixXd(H) - D.transpose() * stiffness.asDiagonal() * D).norm() > 1e-12)
        throw std::runtime_error("Shared-node assembly check failed");
    linalg::AutoSparseSolver linear_solver;
    const auto result = solver::newton_solve(potential, dof::DofReduction(4, {0, 3}), initial,
                                             {.gradient_tolerance = 1e-10}, linear_solver);
    const double tension = (initial[3] - lengths.sum()) / stiffness.cwiseInverse().sum();
    Vec expected = Vec::Zero(4);
    for (int e = 0; e < 3; ++e)
        expected[e + 1] = expected[e] + lengths[e] + tension / stiffness[e];
    if (result.status != solver::NewtonStatus::Converged || (result.x - expected).norm() > 1e-10 ||
        result.x[0] != initial[0] || result.x[3] != initial[3] ||
        potential.eval_gradient(result.x).segment(1, 2).lpNorm<Eigen::Infinity>() > 1e-10)
        throw std::runtime_error("Fixed-end equilibrium check failed");
    std::cout << "custom mesh energy: 3 springs, shared-node assembly and fixed-end equilibrium verified\n";
}
