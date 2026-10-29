#include "litepgo/energy/vertex_pulling_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/io/obj.hpp"
#include "litepgo/io/veg.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/mesh/extract_surface.hpp"
#include "litepgo/solver/newton.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace litepgo;
    const auto mesh = std::get<TetMesh>(io::read_veg(argc > 1 ? argv[1] : LITEPGO_ASSETS_DIR "/box-tet.veg"));
    const auto& topology = mesh.topology;
    const Vec& rest_positions = mesh.positions;
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> rest_nodes(
        rest_positions.data(), topology.num_vertices(), 3);
    if (topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty mesh");
    const VertexDofMap rest_position_map(topology, 3);
    using XField = InterpolatedField3D<TetP1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    auto elastic = std::make_shared<
        VolumetricFemEnergy<NeoHookean, TetP1Basis, TetCentroidQuadrature, XField, IdentityField3D, IdentityField3D>>(
        rest_position_map, std::move(rest_position_fields),
        std::vector<IdentityField3D>(topology.num_cells(), Fp_field),
        std::vector<IdentityField3D>(topology.num_cells(), theta_field));

    // Pull the bounding-box corners toward a copy of the box enlarged about its minimum
    // corner, whose anchor coincides with that corner's rest position.
    const Eigen::Vector3d lower = rest_nodes.colwise().minCoeff().transpose();
    const Eigen::Vector3d upper = rest_nodes.colwise().maxCoeff().transpose();
    const double tolerance = (upper - lower).maxCoeff() * 1e-8;
    if (!((upper - lower).array() > tolerance).all())
        throw std::runtime_error("Example requires nonzero extent along every axis");
    std::vector<int> corners;
    for (int v = 0; v < topology.num_vertices(); ++v) {
        const Eigen::Vector3d X = rest_nodes.row(v).transpose();
        if (((X - lower).array().abs() <= tolerance || (X - upper).array().abs() <= tolerance).all())
            corners.push_back(v);
    }
    if (corners.size() != 8)
        throw std::runtime_error("Example requires a vertex at each bounding-box corner");
    const double stretch = 0.3, stiffness = 1;
    std::vector<std::array<int, 3>> pulled_dofs;
    Vec targets(3 * corners.size());
    for (std::size_t i = 0; i < corners.size(); ++i) {
        const int v = corners[i];
        pulled_dofs.push_back({3 * v, 3 * v + 1, 3 * v + 2});
        targets.segment<3>(3 * Eigen::Index(i)) = stretch * (rest_nodes.row(v).transpose() - lower);
    }
    auto pulling =
        std::make_shared<VertexPullingEnergy>(elastic->num_dofs(), std::move(pulled_dofs), targets, stiffness);

    // Static equilibrium minimizes Pi(u) = E_elastic(u) + E_pulling(u); the springs
    // remove all rigid motions, so no DOF is fixed.
    WeightedEnergySum potential({{elastic, 1}, {pulling, 1}});
    const DofReduction unconstrained(elastic->num_dofs(), {});
    const Vec initial = Vec::Zero(elastic->num_dofs());
    NewtonOptions options;
    options.gradient_tolerance = 1e-8;
    EigenLDLT solver;
    const auto result = newton_solve(potential, unconstrained, initial, options, solver);
    const bool converged = result.status == NewtonStatus::Converged;
    double corner_gap = 0;
    for (std::size_t i = 0; i < corners.size(); ++i)
        corner_gap = std::max(corner_gap,
                              (result.x.segment<3>(3 * corners[i]) - targets.segment<3>(3 * Eigen::Index(i))).norm());
    std::cout << "converged=" << converged << ", iterations=" << result.iterations << ", potential=" << result.energy
              << ", elastic=" << elastic->eval_value(result.x) << ", pulling=" << pulling->eval_value(result.x)
              << ", free residual=" << result.gradient_norm << ", max corner-anchor distance=" << corner_gap << '\n';
    const std::filesystem::path output = argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/tet_p1_mesh_energy.veg";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    const TetMesh deformed{rest_positions + result.x, topology};
    io::write_veg(output, deformed);
    io::write_obj(std::filesystem::path(output).replace_extension(".obj"), extract_surface(deformed));
    return converged ? 0 : 1;
}
