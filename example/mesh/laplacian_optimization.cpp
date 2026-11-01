// Boundary-fixed smoothing and differential-coordinate mesh editing.
// Build: cmake --build build/release --target laplacian_optimization
// Run: build/release/example/mesh/laplacian_optimization

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/solver/newton.hpp"
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace litepgo;
    if (argc > 2)
        throw std::invalid_argument("Usage: laplacian_optimization [output_directory]");
    const std::filesystem::path output = argc == 2 ? argv[1] : "output/cpp/laplacian_optimization";
    constexpr int n = 9, vertices = n * n;
    Vec rest = Vec::Zero(3 * vertices);
    std::vector<int> boundary_dofs, handle_dofs;
    std::vector<mesh::TriangleTopology::Cell> faces;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int a = i * n + j;
            rest[3 * a] = double(j) / (n - 1);
            rest[3 * a + 1] = double(i) / (n - 1);
            for (int c = 0; c < 3; ++c) {
                if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
                    boundary_dofs.push_back(3 * a + c);
                if (j == 0 || j == n - 1)
                    handle_dofs.push_back(3 * a + c);
            }
            if (i < n - 1 && j < n - 1) {
                faces.push_back({a, a + 1, a + n + 1});
                faces.push_back({a, a + n + 1, a + n});
            }
        }
    }
    const mesh::TriangleTopology topology(vertices, faces);
    std::set<std::pair<int, int>> edges;
    for (const auto& face : faces)
        for (int k = 0; k < 3; ++k)
            edges.insert(std::minmax(face[k], face[(k + 1) % 3]));
    std::vector<std::vector<int>> edge_stencils, rings(vertices);
    for (int i = 0; i < vertices; ++i)
        rings[i].push_back(i);
    for (const auto& [i, j] : edges) {
        edge_stencils.push_back({i, j});
        rings[i].push_back(j);
        rings[j].push_back(i);
    }
    const dof::StencilDofMap edge_map(vertices, edge_stencils, 3);
    const dof::StencilDofMap ring_map(vertices, rings, 3);
    Eigen::Matrix<double, 3, 6> B_edge;
    B_edge << Eigen::Matrix3d::Identity(), -Eigen::Matrix3d::Identity();
    energy::MeshEnergy smooth_energy(edge_map, [B_edge](std::span<const int> ids, const energy::BatchMatrix& inputs,
                                                        energy::BatchMatrix* gradients, energy::BatchMatrix* hessians) {
        const Eigen::Index n = inputs.cols();
        Vec values(ids.size());
        if (gradients)
            gradients->resize(ids.size(), n);
        if (hessians)
            hessians->resize(ids.size() * n, n);
        for (Eigen::Index batch = 0; batch < inputs.rows(); ++batch) {
            [[maybe_unused]] const int element = ids[batch];
            const Eigen::MatrixXd x = inputs.row(batch).reshaped<Eigen::RowMajor>(n / 3, 3);

            // One undirected edge: 1/2 ||x_i - x_j||^2.
            const Eigen::Vector3d d = (x.row(0) - x.row(1)).transpose();
            if (gradients)
                gradients->row(batch) = (B_edge.transpose() * d).transpose();
            if (hessians)
                hessians->middleRows(batch * n, n) = B_edge.transpose() * B_edge;

            values[batch] = 0.5 * d.squaredNorm();
        }
        return values;
    });
    Vec noisy = rest;
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            noisy[3 * (i * n + j) + 2] = 0.15 * std::sin(0.7 * i) * std::sin(1.1 * j);
    const solver::NewtonOptions options{.gradient_tolerance = 1e-10};
    linalg::AutoSparseSolver linear_solver;
    // Solve only the free coordinates; prescribed coordinates retain their initial values.
    const auto smooth = solver::newton_solve(smooth_energy, dof::DofReduction(3 * vertices, boundary_dofs), noisy,
                                             options, linear_solver);
    if (smooth.status != solver::NewtonStatus::Converged || (smooth.x - rest).lpNorm<Eigen::Infinity>() > 1e-9 ||
        !(smooth.energy < smooth_energy.eval_value(noisy)))
        throw std::runtime_error("Smoothing check failed");
    for (int d : boundary_dofs)
        if (smooth.x[d] != noisy[d])
            throw std::runtime_error("Smoothing moved a boundary vertex");

    energy::MeshEnergy edit_energy(
        ring_map, [&rest, &ring_map](std::span<const int> ids, const energy::BatchMatrix& inputs,
                                     energy::BatchMatrix* gradients, energy::BatchMatrix* hessians) {
            const Eigen::Index n = inputs.cols();
            Vec values(ids.size());
            if (gradients)
                gradients->resize(ids.size(), n);
            if (hessians)
                hessians->resize(ids.size() * n, n);
            for (Eigen::Index batch = 0; batch < inputs.rows(); ++batch) {
                [[maybe_unused]] const int i = ids[batch];
                const Eigen::MatrixXd x = inputs.row(batch).reshaped<Eigen::RowMajor>(n / 3, 3);

                // B x_local = (LX)_i; the center is the first stencil vertex.
                Eigen::MatrixXd B(3, 3 * x.rows());
                for (Eigen::Index j = 0; j < x.rows(); ++j)
                    B.middleCols<3>(3 * j) = (j == 0 ? double(x.rows() - 1) : -1.) * Eigen::Matrix3d::Identity();
                const Eigen::MatrixXd p = ring_map.gather(i, rest);
                const Eigen::Vector3d r = B * Vec((x - p).reshaped<Eigen::RowMajor>());
                if (gradients)
                    gradients->row(batch) = (B.transpose() * r).transpose();
                if (hessians)
                    hessians->middleRows(batch * n, n) = B.transpose() * B;

                values[batch] = 0.5 * r.squaredNorm();
            }
            return values;
        });
    Vec initial = rest;
    for (int i = 0; i < n; ++i)
        initial[3 * (i * n + n - 1) + 2] = 0.5;
    const auto edited = solver::newton_solve(edit_energy, dof::DofReduction(3 * vertices, handle_dofs), initial,
                                             options, linear_solver);
    const double before = std::sqrt(2. * edit_energy.eval_value(initial)),
                 after = std::sqrt(2. * edit_energy.eval_value(edited.x));
    if (edited.status != solver::NewtonStatus::Converged || !(after < before) ||
        edited.x[3 * (vertices / 2) + 2] <= 0.1)
        throw std::runtime_error("Mesh editing check failed");
    for (int d : handle_dofs)
        if (edited.x[d] != initial[d])
            throw std::runtime_error("Mesh editing moved a prescribed handle");
    for (int a = 0; a < vertices; ++a)
        for (int c = 0; c < 2; ++c)
            if (std::abs(edited.x[3 * a + c] - rest[3 * a + c]) > 1e-9)
                throw std::runtime_error("Mesh editing changed an in-plane coordinate");
    std::filesystem::create_directories(output);
    mesh::write_obj(output / "rest.obj", {rest, topology});
    mesh::write_obj(output / "noisy.obj", {noisy, topology});
    mesh::write_obj(output / "smoothed.obj", {smooth.x, topology});
    mesh::write_obj(output / "handles.obj", {initial, topology});
    mesh::write_obj(output / "edited.obj", {edited.x, topology});
    std::cout << "smoothing: energy=" << smooth.energy << ", free_gradient=" << smooth.gradient_norm
              << "\nediting: differential residual " << before << " -> " << after
              << ", free_gradient=" << edited.gradient_norm << "\nOBJ meshes: " << output << '\n';
}
