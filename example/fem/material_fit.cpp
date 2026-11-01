#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_material_response.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/newton.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "litepgo/response/fixed_pattern_response.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include <Eigen/LU>
#include <cmath>

// Synthetic force calibration; all trial solves start from the same displacement.
int main() {
    using namespace litepgo;
    try {
        const auto mesh = std::get<mesh::TetMesh>(mesh::read_veg(LITEPGO_ASSETS_DIR "/box-tet.veg"));
        const dof::VertexDofMap map(mesh.topology, 3);
        using XField = fem::InterpolatedField3D<fem::TetP1Basis>;
        using Field = fem::IdentityField3D;
        using Elastic = fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature, XField,
                                                 Field, Field>;
        std::vector<XField> geometry;
        for (int e = 0; e < mesh.topology.num_cells(); ++e)
            geometry.emplace_back(map.gather(e, mesh.positions));
        const auto nodes = mesh.positions.reshaped<Eigen::RowMajor>(mesh.topology.num_vertices(), 3);
        const double bottom = nodes.col(2).minCoeff(), top = nodes.col(2).maxCoeff();
        const double height = top - bottom;
        if (!(height > 0))
            throw std::runtime_error("Empty height");
        std::vector<int> fixed_dofs, top_vertices;
        for (int v = 0; v < mesh.topology.num_vertices(); ++v) {
            const bool upper = std::abs(nodes(v, 2) - top) < height * 1e-8;
            if (upper)
                top_vertices.push_back(v);
            if (upper || std::abs(nodes(v, 2) - bottom) < height * 1e-8)
                for (int c = 0; c < 3; ++c)
                    fixed_dofs.push_back(3 * v + c);
        }
        if (fixed_dofs.empty() || fixed_dofs.size() >= static_cast<std::size_t>(mesh.positions.size()))
            throw std::runtime_error("Calibration requires constrained and free nodes");
        const dof::DofReduction fixed(3 * mesh.topology.num_vertices(), fixed_dofs);
        int solves = 0;
        auto reaction = [&](const Eigen::Vector2d& theta, int axis, double strain, Eigen::RowVector2d* derivative) {
            if (!theta.allFinite() || (theta.array() <= 0).any())
                throw std::domain_error("Invalid material parameters");
            Elastic elastic(map, geometry,
                            std::vector<Field>(geometry.size(), Field(Eigen::Matrix3d::Identity().reshaped())),
                            std::vector<Field>(geometry.size(), Field(theta)));
            Vec initial = Vec::Zero(mesh.positions.size());
            for (int v : top_vertices)
                initial[3 * v + axis] = strain * height;
            solver::NewtonOptions options;
            options.gradient_tolerance = 1e-10;
            options.max_iterations = 100;
            linalg::AutoSparseSolver linear_solver;
            ++solves;
            const auto result = solver::newton_solve(elastic, fixed, initial, options, linear_solver);
            if (result.status != solver::NewtonStatus::Converged)
                throw std::domain_error("Forward equilibrium did not converge");
            for (int i : fixed_dofs)
                if (result.x[i] != initial[i])
                    throw std::logic_error("Prescribed displacement changed");
            const Vec gradient = elastic.eval_gradient(result.x);
            double force = 0;
            for (int v : top_vertices)
                force += gradient[3 * v + axis];
            if (derivative) {
                // Partial derivative at fixed equilibrium displacement; the solve below adds du/dtheta.
                using Force = fem::VolumetricFemForceMaterialResponse<fem::NeoHookean, fem::TetP1Basis,
                                                                      fem::TetCentroidQuadrature, XField, Field>;
                const Force response(
                    dof::ConstantDofMap(mesh.topology, 2), map, geometry, result.x,
                    std::vector<Field>(geometry.size(), Field(Eigen::Matrix3d::Identity().reshaped())));
                const Eigen::MatrixXd B = response.eval_jacobian(theta);
                const SpMat K = elastic.eval_hessian(result.x, energy::HessianKind::Exact);
                if (!linear_solver.compute(fixed.reduce(K)))
                    throw std::runtime_error("Sensitivity factorization failed");
                derivative->setZero();
                for (int j = 0; j < 2; ++j) {
                    const Vec rhs = -fixed.reduce(Vec(B.col(j)));
                    Vec sensitivity = Vec::Zero(elastic.num_dofs());
                    fixed.expand(linear_solver.solve(rhs), sensitivity);
                    const Vec force_derivative = B.col(j) + K * sensitivity;
                    if (fixed.reduce(force_derivative).lpNorm<Eigen::Infinity>() > 1e-8)
                        throw std::runtime_error("Sensitivity equation residual too large");
                    for (int v : top_vertices)
                        (*derivative)[j] += force_derivative[3 * v + axis];
                }
            }
            // Uniform stiffness scaling leaves the equilibrium shape unchanged.
            if (derivative && std::abs(derivative->dot(theta) - force) > 1e-7 * std::max(1.0, std::abs(force)))
                throw std::runtime_error("Reaction sensitivity violates stiffness scaling");
            return force;
        };
        const Eigen::Vector2d truth(2, 3);
        auto measurements = [&](const Eigen::Vector2d& theta, Eigen::MatrixXd* J) {
            Eigen::RowVector2d shear, compression;
            const Eigen::Vector2d forces(reaction(theta, 0, 0.08, J ? &shear : nullptr),
                                         reaction(theta, 2, -0.04, J ? &compression : nullptr));
            if (J) {
                J->resize(2, 2);
                J->row(0) = shear;
                J->row(1) = compression;
            }
            return forces;
        };
        const Eigen::Vector2d observed = measurements(truth, nullptr);
        const Eigen::Vector2d scales = observed.cwiseAbs().cwiseMax(1e-8);
        class MaterialResidual final : public response::FixedPatternResponse {
          public:
            MaterialResidual(const decltype(measurements)& experiments, Eigen::Vector2d target, Eigen::Vector2d scale)
                : FixedPatternResponse(Eigen::Matrix2d::Ones().sparseView()), measurements_(experiments),
                  observed(target), scales(scale) {}

          private:
            Vec eval_values(const ConstVecRef& q, SpMat* J) const override {
                Eigen::MatrixXd dense_J;
                const Eigen::Vector2d theta = q.array().exp().matrix();
                const Eigen::Vector2d forces = measurements_(theta, J ? &dense_J : nullptr);
                if (J) {
                    dense_J = (scales.cwiseInverse().asDiagonal() * dense_J * theta.asDiagonal()).eval();
                    for (int c = 0; c < 2; ++c)
                        for (int r = 0; r < 2; ++r)
                            J->coeffRef(r, c) = dense_J(r, c);
                }
                return (forces - observed).cwiseQuotient(scales);
            }
            const decltype(measurements)& measurements_;
            Eigen::Vector2d observed, scales;
        };
        MaterialResidual residual(measurements, observed, scales);
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-8;
        const auto result = solver::gauss_newton_solve(residual, Vec::Zero(2), options);
        const Eigen::Vector2d fitted = result.x.array().exp().matrix();
        const Vec r = residual.eval_value(result.x);
        const double parameter_error = (fitted - truth).norm() / truth.norm();
        const double reference = reaction(truth, 2, -0.07, nullptr);
        const double prediction_error = std::abs(reaction(fitted, 2, -0.07, nullptr) - reference) / std::abs(reference);
        std::cout << "truth=" << truth.transpose() << ", fitted=" << fitted.transpose()
                  << ", parameter error=" << parameter_error << ", residual=" << r.norm()
                  << ", held-out error=" << prediction_error << ", forward solves=" << solves << '\n';
        return result.status == solver::GaussNewtonStatus::Converged && parameter_error < 1e-4 && r.norm() < 1e-7 &&
                       prediction_error < 1e-4
                   ? 0
                   : 1;
    } catch (const std::exception& error) {
        std::cerr << "material_fit: " << error.what() << '\n';
        return 1;
    }
}
