#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>
// pull a tet box's corners and solve static equilibrium.
// Build: cmake --build build/release --target tet_p1_corner_pull
// Run: build/release/examples/fem/equilibrium/tet_p1_corner_pull

#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/energy/scalar/vertex_pulling_energy.hpp"
#include "litepgo/energy/scalar/weighted_sum_energy.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/extract_surface.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/newton.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace litepgo;
    const auto mesh = std::get<mesh::TetMesh>(mesh::read_veg(argc > 1 ? argv[1] : LITEPGO_ASSETS_DIR "/box-tet.veg"));
    const auto& topology = mesh.topology;
    const Vec& rest_positions = mesh.positions;
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> rest_nodes(
        rest_positions.data(), topology.num_vertices(), 3);
    if (topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty mesh");
    const dof::VertexDofMap rest_position_map(topology, 3);

    std::vector<std::shared_ptr<litepgo::energy::Response>> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.push_back(
            std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(rest_position_map.gather(e, rest_positions)));
    auto Fp_field =
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(9, 3), Eigen::Matrix3d::Identity().reshaped());
    auto theta_field = std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(2, 3));
    auto elastic = std::make_shared<fem::VolumetricFemEnergy<fem::TetP1Basis, fem::TetCentroidQuadrature>>(
        std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<fem::NeoHookean>>(), rest_position_map,
        std::move(rest_position_fields),
        std::vector<std::shared_ptr<litepgo::energy::Response>>(topology.num_cells(), Fp_field),
        std::vector<std::shared_ptr<litepgo::energy::Response>>(topology.num_cells(), theta_field));

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
        std::make_shared<energy::VertexPullingEnergy>(elastic->num_dofs(), std::move(pulled_dofs), targets, stiffness);

    // Static equilibrium minimizes Pi(u) = E_elastic(u) + E_pulling(u); the springs
    // remove all rigid motions, so no DOF is fixed.
    energy::WeightedSumEnergy potential({{elastic, 1}, {pulling, 1}});
    const dof::DofReduction unconstrained(elastic->num_dofs(), {});
    const Vec initial = Vec::Zero(elastic->num_dofs());
    solver::NewtonOptions options;

    linalg::AutoSparseSolver solver;
    const auto result = solver::newton_solve(potential, unconstrained, initial, options, solver,
                                             litepgo::solver::AbsoluteConvergence(1e-8));
    const bool converged = result.status == solver::NewtonStatus::Converged;
    double corner_gap = 0;
    for (std::size_t i = 0; i < corners.size(); ++i)
        corner_gap = std::max(corner_gap,
                              (result.x.segment<3>(3 * corners[i]) - targets.segment<3>(3 * Eigen::Index(i))).norm());
    std::cout << "converged=" << converged << ", iterations=" << result.iterations << ", potential=" << result.energy
              << ", elastic=" << elastic->evaluate(result.x)->value()
              << ", pulling=" << pulling->evaluate(result.x)->value() << ", free residual=" << result.gradient_norm
              << ", max corner-anchor distance=" << corner_gap << '\n';
    const std::filesystem::path output = argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/tet_p1_corner_pull.veg";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    const mesh::TetMesh deformed{rest_positions + result.x, topology};
    mesh::write_veg(output, deformed);
    mesh::write_obj(std::filesystem::path(output).replace_extension(".obj"), mesh::extract_surface(deformed));
    return converged ? 0 : 1;
}
