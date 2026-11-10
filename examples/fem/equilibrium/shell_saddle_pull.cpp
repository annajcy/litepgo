#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include <memory>
// pull a sheet's corners into a saddle and solve static equilibrium.
// Build: cmake --build build/release --target shell_saddle_pull
// Run: build/release/examples/fem/equilibrium/shell_saddle_pull

#include "litepgo/energy/scalar/vertex_pulling_energy.hpp"
#include "litepgo/energy/scalar/weighted_sum_energy.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/newton.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace litepgo;
    const auto mesh = mesh::read_obj(argc > 1 ? argv[1] : LITEPGO_ASSETS_DIR "/shell.obj");
    const auto& topology = mesh.topology;
    const Vec& rest_positions = mesh.positions;
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> rest_nodes(
        rest_positions.data(), topology.num_vertices(), 3);
    if (topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty mesh");
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
    const dof::StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    std::vector<fem::TriangleMidEdgeRestData> rest;
    for (int f = 0; f < map.num_elements(); ++f)
        rest.emplace_back(map.gather(f, rest_positions), stencils.opposite[static_cast<std::size_t>(f)]);

    const double thickness = 0.1, Y = 100, nu = 0.3;
    // Plane-stress Lame parameters for a shell.
    const Eigen::Vector2d theta(Y * nu / (1 - nu * nu), Y / (2 * (1 + nu)));
    auto elastic = std::make_shared<fem::ShellFemEnergy>(
        std::make_shared<litepgo::fem::PointwiseShellAdapter<fem::KoiterStVK>>(), map, std::move(rest),
        std::vector<double>(map.num_elements(), thickness), std::vector<Eigen::VectorXd>(map.num_elements(), theta));

    // Pull the four corners of the flat xy-plane sheet outward in plane and alternately
    // up and down out of plane, so the anchors span a saddle.
    const Eigen::Vector3d lower = rest_nodes.colwise().minCoeff().transpose();
    const Eigen::Vector3d upper = rest_nodes.colwise().maxCoeff().transpose();
    const Eigen::Vector3d center = (lower + upper) / 2;
    const double size = (upper - lower).head<2>().maxCoeff(), tolerance = size * 1e-8;
    if (!((upper - lower).head<2>().array() > tolerance).all() || upper.z() - lower.z() > tolerance)
        throw std::runtime_error("Example requires a flat sheet in the xy plane");
    std::vector<int> corners;
    for (int v = 0; v < topology.num_vertices(); ++v) {
        const Eigen::Vector2d X = rest_nodes.row(v).head<2>().transpose();
        if (((X - lower.head<2>()).array().abs() <= tolerance || (X - upper.head<2>()).array().abs() <= tolerance)
                .all())
            corners.push_back(v);
    }
    if (corners.size() != 4)
        throw std::runtime_error("Example requires a vertex at each bounding-box corner");
    const double stretch = 0.1, lift = 0.2 * size, stiffness = 10;
    std::vector<std::array<int, 3>> pulled_dofs;
    Vec targets(3 * corners.size());
    for (std::size_t i = 0; i < corners.size(); ++i) {
        const int v = corners[i];
        const Eigen::Vector3d offset = rest_nodes.row(v).transpose() - center;
        pulled_dofs.push_back({3 * v, 3 * v + 1, 3 * v + 2});
        targets.segment<3>(3 * Eigen::Index(i)) =
            Eigen::Vector3d(stretch * offset.x(), stretch * offset.y(), offset.x() * offset.y() > 0 ? lift : -lift);
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
    const std::filesystem::path output = argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/shell_saddle_pull.obj";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    mesh::write_obj(output, mesh::TriangleMesh{rest_positions + result.x, topology});
    return converged ? 0 : 1;
}
