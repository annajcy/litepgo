#include "litepgo/energy/quadratic_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/newton.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main() {
    using namespace litepgo;
    // A flat patch with face areas 1, 2, 1/2, 2 and consistent winding.
    Eigen::Matrix<double, 6, 3> rest_nodes;
    rest_nodes << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, -1, 0, 0, 0, -2, 0;
    const Vec rest_positions = rest_nodes.reshaped<Eigen::RowMajor>();

    // Face 0 has three neighbors; boundary faces each have one neighbor.
    const auto topology = TriangleTopology(6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}});
    const auto stencils = make_triangle_mid_edge_stencils(topology);
    const StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    std::vector<TriangleMidEdgeRestData> rest;
    for (int f = 0; f < map.num_elements(); ++f)
        rest.emplace_back(map.gather(f, rest_positions), stencils.opposite[static_cast<std::size_t>(f)]);

    const double thickness = 0.1, Y = 100, nu = 0.3;
    // Plane-stress Lame parameters for a shell.
    const Eigen::Vector2d theta(Y * nu / (1 - nu * nu), Y / (2 * (1 + nu)));
    auto elastic = std::make_shared<ShellFemEnergy<KoiterStVK>>(
        map, std::move(rest), std::vector<double>(map.num_elements(), thickness),
        std::vector<Eigen::VectorXd>(map.num_elements(), theta));

    // Fix the central triangle and pull vertex 3 downward.
    std::vector<int> fixed_dofs;
    for (int v : {0, 1, 2})
        for (int c = 0; c < 3; ++c)
            fixed_dofs.push_back(3 * v + c);
    Vec force = Vec::Zero(elastic->num_dofs());
    force[3 * 3 + 2] = -0.001;

    // Static equilibrium minimizes total potential: Pi(u) = E_elastic(u) - force.dot(u).
    SpMat zero(elastic->num_dofs(), elastic->num_dofs());
    auto load = std::make_shared<QuadraticEnergy>(zero, -force);
    WeightedEnergySum potential({{elastic, 1}, {load, 1}});
    const DofReduction fixed(elastic->num_dofs(), std::move(fixed_dofs));
    const Vec initial = Vec::Zero(elastic->num_dofs());
    NewtonOptions options;
    options.gradient_tolerance = 1e-8;
    EigenLDLT solver;
    const auto result = newton_solve(potential, fixed, initial, options, solver);
    const bool converged = result.status == NewtonStatus::Converged;
    std::cout << "converged=" << converged << ", iterations=" << result.iterations << ", potential=" << result.energy
              << ", elastic=" << elastic->eval_value(result.x) << ", free residual=" << result.gradient_norm
              << ", tip displacement=" << result.x.segment<3>(3 * 3).transpose() << '\n';
    return converged ? 0 : 1;
}
