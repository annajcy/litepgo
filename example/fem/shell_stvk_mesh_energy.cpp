#include "litepgo/energy/quadratic_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/io/obj.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/newton.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace litepgo;
    const auto mesh = io::read_obj(argc > 1 ? argv[1] : LITEPGO_ASSETS_DIR "/shell.obj");
    const auto& topology = mesh.topology;
    const Vec& rest_positions = mesh.positions;
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> rest_nodes(
        rest_positions.data(), topology.num_vertices(), 3);
    if (topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty mesh");
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
            force[3 * v + 2] = -0.001 / loaded;
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
    const std::filesystem::path output = argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/shell_stvk_mesh_energy.obj";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    io::write_obj(output, TriangleMesh{rest_positions + result.x, topology});
    return converged ? 0 : 1;
}
