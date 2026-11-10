// MeshEnergy subclass assembly for a spring chain with fixed endpoints.
// Build: cmake --build build/release --target mesh_energy_subclass
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/solver/newton.hpp"
#include <iostream>
#include <stdexcept>

using namespace litepgo;
class SpringEnergy final : public energy::MeshEnergy {
  public:
    SpringEnergy(const dof::StencilDofMap& map, Eigen::Vector3d k, Eigen::Vector3d l)
        : MeshEnergy(map, 2), stiffness(std::move(k)), lengths(std::move(l)) {}

  private:
    Vec eval_batch(std::span<const int> ids, const energy::BatchMatrix& x, energy::BatchMatrix* g,
                   energy::BatchMatrix* H) const override {
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
    }
    Eigen::Vector3d stiffness, lengths;
};

int main() {
    const dof::StencilDofMap mapping(4, {{0, 1}, {1, 2}, {2, 3}}, 1);
    const Eigen::Vector3d stiffness(2, 3, 4), lengths(1, 2, 1);
    SpringEnergy potential(mapping, stiffness, lengths);
    // The polynomial has no restricted domain, so the default feasible step is one.
    Vec initial(4);
    initial << 0, 0, 0, 7;
    linalg::AutoSparseSolver linear_solver;
    const auto result = solver::newton_solve(potential, dof::DofReduction(4, {0, 3}), initial, {}, linear_solver,
                                             litepgo::solver::AbsoluteConvergence(1e-10));
    if (result.status != solver::NewtonStatus::Converged)
        throw std::runtime_error("Fixed-end equilibrium failed");
    std::cout << "mesh energy subclass: 3 springs, shared-node assembly and fixed-end equilibrium\n";
}
