// Uniform-weight vertex ARAP, alternating local rotations and global positions.
// Build: cmake --build build/release --target arap
// Run: build/release/examples/mesh/arap

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/solver/newton.hpp"
#include <Eigen/LU>
#include <Eigen/SVD>

#include <cmath>
#include <iostream>
#include <numbers>
#include <set>
#include <stdexcept>

using namespace litepgo;

namespace {
std::vector<Eigen::Matrix3d> rotations(const Vec& x, const Vec& rest, const std::set<std::pair<int, int>>& edges) {
    const int vertices = static_cast<int>(rest.size() / 3);
    std::vector<Eigen::Matrix3d> covariance(vertices, Eigen::Matrix3d::Zero());
    for (const auto& [a, b] : edges) {
        const Eigen::Matrix3d outer =
            (x.segment<3>(3 * a) - x.segment<3>(3 * b)) * (rest.segment<3>(3 * a) - rest.segment<3>(3 * b)).transpose();
        covariance[a] += outer;
        covariance[b] += outer;
    }
    for (auto& matrix : covariance) {
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
        // Enforce det(R) = +1: the local fit must not reflect the mesh.
        correction(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
        matrix = svd.matrixU() * correction * svd.matrixV().transpose();
    }
    return covariance;
}

double objective(const Vec& x, const Vec& rest, const std::set<std::pair<int, int>>& edges,
                 const std::vector<Eigen::Matrix3d>& rotation) {
    double value = 0.;
    for (const auto& [a, b] : edges) {
        const Eigen::Vector3d p = rest.segment<3>(3 * a) - rest.segment<3>(3 * b);
        const Eigen::Vector3d d = x.segment<3>(3 * a) - x.segment<3>(3 * b);
        value += 0.25 * ((d - rotation[a] * p).squaredNorm() + (d - rotation[b] * p).squaredNorm());
    }
    return value;
}
class ArapGlobalEnergy final : public energy::MeshEnergy {
  public:
    ArapGlobalEnergy(const dof::StencilDofMap& map, const std::vector<std::vector<int>>& stencils, const Vec& reference,
                     const std::vector<Eigen::Matrix3d>& rotations, const Eigen::Matrix<double, 3, 6>& derivative)
        : MeshEnergy(map), edge_stencils(stencils), rest(reference), rotation(rotations), B(derivative) {}

  private:
    // Borrowed state remains fixed throughout each global solve.
    const std::vector<std::vector<int>>& edge_stencils;
    const Vec& rest;
    const std::vector<Eigen::Matrix3d>& rotation;
    Eigen::Matrix<double, 3, 6> B;
    Vec eval_batch(std::span<const int> ids, const energy::BatchMatrix& inputs, energy::BatchMatrix* gradients,
                   energy::BatchMatrix* hessians) const override {
        const Eigen::Index n = inputs.cols();
        Vec values(ids.size());
        if (gradients)
            gradients->resize(ids.size(), n);
        if (hessians)
            hessians->resize(ids.size() * n, n);
        for (Eigen::Index batch = 0; batch < inputs.rows(); ++batch) {
            [[maybe_unused]] const int e = ids[batch];
            const Eigen::MatrixXd local_x = inputs.row(batch).reshaped<Eigen::RowMajor>(n / 3, 3);

            const auto& stencil = edge_stencils[e];
            const int i = stencil[0], j = stencil[1];
            const Eigen::Vector3d p = rest.segment<3>(3 * i) - rest.segment<3>(3 * j);
            const Eigen::Vector3d d = (local_x.row(0) - local_x.row(1)).transpose();
            const Eigen::Vector3d ri = d - rotation[i] * p, rj = d - rotation[j] * p;
            // Two directed contributions, each with weight 1/4.
            if (gradients)
                gradients->row(batch) = (B.transpose() * (0.5 * (ri + rj))).transpose();
            if (hessians)
                hessians->middleRows(batch * n, n) = B.transpose() * B;

            values[batch] = 0.25 * (ri.squaredNorm() + rj.squaredNorm());
        }
        return values;
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc > 2)
        throw std::invalid_argument("Usage: arap [output_directory]");
    const std::filesystem::path output = argc == 2 ? argv[1] : "output/cpp/arap";
    constexpr int n = 9, vertices = n * n;
    Vec rest = Vec::Zero(3 * vertices);
    std::vector<int> handle_dofs;
    std::vector<mesh::TriangleTopology::Cell> faces;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int a = i * n + j;
            rest[3 * a] = double(j) / (n - 1);
            rest[3 * a + 1] = double(i) / (n - 1);
            for (int c = 0; c < 3; ++c) {
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
    std::vector<std::vector<int>> edge_stencils;
    for (const auto& [i, j] : edges)
        edge_stencils.push_back({i, j});
    const dof::StencilDofMap edge_map(vertices, edge_stencils, 3);
    Eigen::Matrix<double, 3, 6> B;
    B << Eigen::Matrix3d::Identity(), -Eigen::Matrix3d::Identity();
    const double angle = std::numbers::pi / 3.;
    Eigen::Matrix3d R;
    R << std::cos(angle), 0, std::sin(angle), 0, 1, 0, -std::sin(angle), 0, std::cos(angle);
    Vec rigid = rest;
    for (int a = 0; a < vertices; ++a)
        rigid.segment<3>(3 * a) = R * rest.segment<3>(3 * a) + Eigen::Vector3d(0.3, -0.2, 0.7);
    if (!(objective(rigid, rest, edges, rotations(rigid, rest, edges)) < 1e-24))
        throw std::runtime_error("Rigid motion invariance check failed");
    Vec x = rest;
    for (int i = 0; i < n; ++i) {
        const int a = i * n + n - 1;
        x.segment<3>(3 * a) =
            R * (rest.segment<3>(3 * a) - Eigen::Vector3d(1., 0.5, 0.)) + Eigen::Vector3d(0.8, 0.5, 0.5);
    }
    const Vec initial = x;
    // Fixed DOF values come from x, not from DofReduction.
    const dof::DofReduction fixed(3 * vertices, handle_dofs);
    linalg::AutoSparseSolver linear_solver;
    auto rotation = rotations(x, rest, edges);
    const double first = objective(x, rest, edges, rotation);
    double previous = first, change = 0.;
    int iterations = 0;
    for (; iterations < 1000;) {
        // Rotations stay fixed throughout this global solve.
        ArapGlobalEnergy global(edge_map, edge_stencils, rest, rotation, B);
        auto result =
            solver::newton_solve(global, fixed, x, {}, linear_solver, litepgo::solver::AbsoluteConvergence(1e-10));
        if (result.status != solver::NewtonStatus::Converged)
            throw std::runtime_error("ARAP global solve failed");
        // Local step: refit rotations before measuring the full ARAP energy.
        rotation = rotations(result.x, rest, edges);
        const double value = objective(result.x, rest, edges, rotation);
        if (!(value <= previous + 1e-12))
            throw std::runtime_error("ARAP energy increased");
        change = (result.x - x).lpNorm<Eigen::Infinity>();
        x = result.x;
        previous = value;
        ++iterations;
        if (change < 1e-8)
            break;
    }
    if (!(change < 1e-8) || !(previous < first) ||
        (x.segment<3>(3 * (vertices / 2)) - rest.segment<3>(3 * (vertices / 2))).norm() <= 0.05)
        throw std::runtime_error("ARAP convergence check failed");
    for (int d : handle_dofs)
        if (x[d] != initial[d])
            throw std::runtime_error("ARAP moved a prescribed handle");
    std::filesystem::create_directories(output);
    mesh::write_obj(output / "rest.obj", {rest, topology});
    mesh::write_obj(output / "handles.obj", {initial, topology});
    mesh::write_obj(output / "edited.obj", {x, topology});
    std::cout << "ARAP: " << iterations << " iterations, energy " << first << " -> " << previous
              << ", max update=" << change << "\nOBJ meshes: " << output << '\n';
}
