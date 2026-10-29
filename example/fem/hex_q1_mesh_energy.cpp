#include "litepgo/energy/quadratic_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/io/veg.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/solver/newton.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace litepgo;
    const auto mesh = std::get<HexMesh>(io::read_veg(argc > 1 ? argv[1] : LITEPGO_ASSETS_DIR "/box-hex.veg"));
    const auto& topology = mesh.topology;
    const Vec& rest_positions = mesh.positions;
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> rest_nodes(
        rest_positions.data(), topology.num_vertices(), 3);
    if (topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty mesh");
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

    // Clamp the left face and distribute a total downward load over the right face.
    const double xmin = rest_nodes.col(0).minCoeff(), xmax = rest_nodes.col(0).maxCoeff();
    const double tolerance = (xmax - xmin) * 1e-8;
    if (!(xmax > xmin))
        throw std::runtime_error("Example requires nonzero x extent");
    int loaded = 0, tip = 0;
    for (int v = 0; v < topology.num_vertices(); ++v)
        if (rest_nodes(v, 0) >= xmax - tolerance) {
            ++loaded;
            tip = v;
        }
    std::vector<int> fixed_dofs;
    Vec force = Vec::Zero(elastic->num_dofs());
    for (int v = 0; v < topology.num_vertices(); ++v) {
        if (rest_nodes(v, 0) <= xmin + tolerance)
            for (int c = 0; c < 3; ++c)
                fixed_dofs.push_back(3 * v + c);
        if (rest_nodes(v, 0) >= xmax - tolerance)
            force[3 * v + 2] = -0.02 / loaded;
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
              << ", tip displacement=" << result.x.segment<3>(3 * tip).transpose() << '\n';
    const std::filesystem::path output = argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/hex_q1_mesh_energy.veg";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    io::write_veg(output, HexMesh{rest_positions + result.x, topology});
    return converged ? 0 : 1;
}
