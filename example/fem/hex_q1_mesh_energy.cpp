#include "litepgo/energy/quadratic_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/solver/newton.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main() {
    using namespace litepgo;
    // Two adjacent hexes of widths 1 and 2, sharing the face at X = 1.
    Eigen::Matrix<double, 12, 3> rest_nodes;
    rest_nodes << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 3, 0, 0, 3, 1, 0, 3, 0, 1, 3,
        1, 1;
    const Vec rest_positions = rest_nodes.reshaped<Eigen::RowMajor>();

    const HexTopology topology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
    const VertexDofMap rest_position_map(topology, 3);
    using XField = InterpolatedField3D<HexQ1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    auto elastic = std::make_shared<
        VolumetricFemEnergy<NeoHookean, HexQ1Basis, GaussLegendre3D<2>, XField, IdentityField3D, IdentityField3D>>(
        rest_position_map, std::move(rest_position_fields),
        std::vector<IdentityField3D>(topology.num_cells(), Fp_field),
        std::vector<IdentityField3D>(topology.num_cells(), theta_field));

    // Clamp X=0; distribute a total downward force of 0.02 over X=3.
    std::vector<int> fixed_dofs;
    Vec force = Vec::Zero(elastic->num_dofs());
    for (int v = 0; v < topology.num_vertices(); ++v) {
        if (rest_nodes(v, 0) == 0)
            for (int c = 0; c < 3; ++c)
                fixed_dofs.push_back(3 * v + c);
        if (rest_nodes(v, 0) == 3)
            force[3 * v + 2] = -0.02 / 4;
    }

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
              << ", tip displacement=" << result.x.segment<3>(3 * 8).transpose() << '\n';
    return converged ? 0 : 1;
}
