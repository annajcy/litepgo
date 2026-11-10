#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include <memory>
// Differential plastic growth of a clamped shell strip; solve u and write OBJ.
// Build: cmake --build build/release --target shell_plastic_field
// Run: build/release/examples/fem/plasticity/shell_plastic_field

#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/newton.hpp"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    using namespace litepgo;
    try {
        constexpr int nx = 12, ny = 4, count = (nx + 1) * (ny + 1), fixed_count = 3 * 2 * (ny + 1);
        Vec X(3 * count);
        for (int i = 0; i <= nx; ++i)
            for (int j = 0; j <= ny; ++j)
                X.segment<3>(3 * (i * (ny + 1) + j)) = Eigen::Vector3d(4.0 * i / nx, double(j) / ny - .5, 0);
        std::vector<std::array<int, 3>> faces;
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j) {
                const int a = i * (ny + 1) + j, b = a + ny + 1;
                faces.push_back({a, b, b + 1});
                faces.push_back({a, b + 1, a + 1});
            }
        const mesh::TriangleTopology topology(count, faces);
        const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
        const dof::StencilDofMap map(count, stencils.vertices, 3);
        std::vector<fem::TriangleMidEdgeRestData> rest;
        for (int e = 0; e < map.num_elements(); ++e)
            rest.emplace_back(map.gather(e, X), stencils.opposite[e]);
        const std::vector<double> thickness(rest.size(), .2);
        const std::vector<Vec> theta(rest.size(), Eigen::Vector2d(2, 3));
        // Two vertex columns clamp both position and the direction of the strip.
        std::vector<int> fixed_dofs;
        for (int i = 0; i < fixed_count; ++i)
            fixed_dofs.push_back(i);
        const dof::DofReduction fixed(map.num_dofs(), fixed_dofs);
        Vec u = Vec::Zero(map.num_dofs());
        solver::NewtonOptions options;

        options.max_iterations = 100;
        const std::filesystem::path output = argc > 1 ? argv[1] : LITEPGO_OUTPUT_DIR "/shell_plastic_field";
        std::filesystem::create_directories(output);
        mesh::write_obj(output / "rest.obj", mesh::TriangleMesh{X, topology});
        double max_displacement = 0;
        for (int step = 0; step <= 4; ++step) {
            const double alpha = .04 * step / 4;
            std::vector<Eigen::Matrix3d> plastic;
            for (const auto& r : rest)
                plastic.push_back(
                    Eigen::Vector3d(std::exp(alpha * 2 * r.X_local.topRows(3).col(1).mean()), 1, 1).asDiagonal());
            fem::ShellFemEnergy elastic(std::make_shared<litepgo::fem::PointwiseShellAdapter<fem::KoiterStVK>>(), map,
                                        rest, thickness, theta, plastic);
            linalg::AutoSparseSolver linear;
            const auto result =
                solver::newton_solve(elastic, fixed, u, options, linear, litepgo::solver::AbsoluteConvergence(1e-9));
            if (result.status != solver::NewtonStatus::Converged)
                throw std::runtime_error("Static equilibrium failed at step " + std::to_string(step));
            u = result.x;
            const double residual = elastic.evaluate(u)->gradient().tail(u.size() - fixed_count).norm();
            max_displacement = 0;
            for (int v = 0; v < count; ++v)
                max_displacement = std::max(max_displacement, u.segment<3>(3 * v).norm());
            std::cout << std::setprecision(12) << "step=" << step << ", energy=" << result.energy
                      << ", free_residual=" << residual << ", max_displacement=" << max_displacement << '\n';
            if (residual > 1e-8 || u.head(fixed_count).cwiseAbs().maxCoeff() > 1e-14)
                throw std::runtime_error("Equilibrium or clamp check failed");
            if (step == 0 && max_displacement > 1e-12)
                throw std::runtime_error("Identity plastic field changed the rest pose");
        }
        if (max_displacement < .01)
            throw std::runtime_error("Plastic field did not deform the strip");
        mesh::write_obj(output / "equilibrium.obj", mesh::TriangleMesh{X + u, topology});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shell_plastic_field: " << error.what() << '\n';
        return 1;
    }
}
