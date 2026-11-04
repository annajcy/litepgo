// Fit volume material parameters from reactions using analytic equilibrium sensitivities.
// Build: cmake --build build/release --target volume_equilibrium_material_fit
// Run: build/release/examples/fem/fitting/volume_equilibrium_material_fit

#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_material_response.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/newton.hpp"
#include "litepgo/solver/static_equilibrium_response.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include <cmath>

#include <functional>
namespace {
using namespace litepgo;
using XField = fem::InterpolatedField3D<fem::TetP1Basis>;
using Field = fem::IdentityField3D;
using Elastic =
    fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature, XField, Field, Field>;
struct Problem {
    dof::VertexDofMap map;
    std::vector<XField> geometry;
    dof::ConstantDofMap parameter_map;
    dof::DofReduction fixed;
    std::vector<int> top_vertices;
    double height;
    mutable int solves = 0;
};
Problem make_problem() {
    const auto volume = std::get<mesh::TetMesh>(mesh::read_veg(LITEPGO_ASSETS_DIR "/box-tet.veg"));
    const dof::VertexDofMap map(volume.topology, 3);
    std::vector<XField> geometry;
    for (int e = 0; e < volume.topology.num_cells(); ++e)
        geometry.emplace_back(map.gather(e, volume.positions));
    const auto nodes = volume.positions.reshaped<Eigen::RowMajor>(volume.topology.num_vertices(), 3);
    const double bottom = nodes.col(2).minCoeff(), top = nodes.col(2).maxCoeff();
    const double height = top - bottom;
    if (!(height > 0))
        throw std::runtime_error("Empty height");
    std::vector<int> fixed_dofs, top_vertices;
    for (int v = 0; v < volume.topology.num_vertices(); ++v) {
        const bool upper = std::abs(nodes(v, 2) - top) < height * 1e-8;
        if (upper)
            top_vertices.push_back(v);
        if (upper || std::abs(nodes(v, 2) - bottom) < height * 1e-8)
            for (int c = 0; c < 3; ++c)
                fixed_dofs.push_back(3 * v + c);
    }
    if (fixed_dofs.empty() || fixed_dofs.size() >= static_cast<std::size_t>(volume.positions.size()))
        throw std::runtime_error("Calibration requires constrained and free nodes");
    const dof::DofReduction fixed(3 * volume.topology.num_vertices(), fixed_dofs);
    return {map, geometry, dof::ConstantDofMap(volume.topology, 2), fixed, top_vertices, height};
}
double reaction(const Problem& problem, const Eigen::Vector2d& theta, int axis, double strain,
                Eigen::RowVector2d* derivative) {
    const auto& map = problem.map;
    const auto& geometry = problem.geometry;
    const auto& top_vertices = problem.top_vertices;
    const auto& fixed = problem.fixed;
    const double height = problem.height;
    if (!theta.allFinite() || (theta.array() <= 0).any())
        throw std::domain_error("Invalid material parameters");
    Elastic elastic(map, geometry, std::vector<Field>(geometry.size(), Field(Eigen::Matrix3d::Identity().reshaped())),
                    std::vector<Field>(geometry.size(), Field(theta)));
    Vec initial = Vec::Zero(map.num_dofs());
    for (int v : top_vertices)
        initial[3 * v + axis] = strain * height;
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-10;
    options.max_iterations = 100;
    // Partial force derivative at fixed displacement; equilibrium supplies du/dp separately.
    const auto make_response = [&](const Vec& displacement) {
        using Force = fem::VolumetricFemForceMaterialResponse<fem::NeoHookean, fem::TetP1Basis,
                                                              fem::TetCentroidQuadrature, XField, Field>;
        return std::make_shared<Force>(
            problem.parameter_map, map, geometry, displacement,
            std::vector<Field>(geometry.size(), Field(Eigen::Matrix3d::Identity().reshaped())));
    };

    solver::StaticEquilibriumResponse equilibrium(
        2, fixed, initial, options,
        [&](const Vec& parameters) -> std::shared_ptr<energy::Energy> {
            ++problem.solves;
            return std::make_shared<Elastic>(
                map, geometry, std::vector<Field>(geometry.size(), Field(Eigen::Matrix3d::Identity().reshaped())),
                std::vector<Field>(geometry.size(), Field(parameters)));
        },
        make_response);
    SpMat J = equilibrium.jacobian_pattern();
    const Vec displacement = equilibrium.eval(theta, derivative ? &J : nullptr);
    const Vec gradient = elastic.eval_gradient(displacement);
    double force = 0;
    for (int v : top_vertices)
        force += gradient[3 * v + axis];
    if (derivative) {
        const Eigen::MatrixXd force_derivative =
            Eigen::MatrixXd(make_response(displacement)->eval_jacobian(theta)) + elastic.eval_hessian(displacement) * J;
        derivative->setZero();
        for (int v : top_vertices)
            *derivative += force_derivative.row(3 * v + axis);
    }
    return force;
}
class MaterialResidual final : public energy::FixedPatternResponse {
  public:
    MaterialResidual(const std::function<Eigen::Vector2d(const Eigen::Vector2d&, Eigen::MatrixXd*)>& experiments,
                     Eigen::Vector2d target, Eigen::Vector2d scale)
        : FixedPatternResponse(Eigen::Matrix2d::Ones().sparseView()), measurements_(experiments), observed(target),
          scales(scale) {}

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
    std::function<Eigen::Vector2d(const Eigen::Vector2d&, Eigen::MatrixXd*)> measurements_;
    Eigen::Vector2d observed, scales;
};
} // namespace
int main() {
    using namespace litepgo;
    try {
        const auto problem = make_problem();
        const Eigen::Vector2d truth(2, 3);
        auto measurements = [&](const Eigen::Vector2d& theta, Eigen::MatrixXd* J) {
            Eigen::RowVector2d shear, compression;
            const Eigen::Vector2d forces(reaction(problem, theta, 0, 0.08, J ? &shear : nullptr),
                                         reaction(problem, theta, 2, -0.04, J ? &compression : nullptr));
            if (J) {
                J->resize(2, 2);
                J->row(0) = shear;
                J->row(1) = compression;
            }
            return forces;
        };
        const Eigen::Vector2d observed = measurements(truth, nullptr);
        const Eigen::Vector2d scales = observed.cwiseAbs().cwiseMax(1e-8);
        MaterialResidual residual(measurements, observed, scales);
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-8;
        // Fit the measured response using its analytic parameter Jacobian.
        const auto result = solver::gauss_newton_solve(residual, Vec::Zero(2), options);
        const Eigen::Vector2d fitted = result.x.array().exp().matrix();
        const Vec r = residual.eval_value(result.x);
        const double parameter_error = (fitted - truth).norm() / truth.norm();
        const double reference = reaction(problem, truth, 2, -0.07, nullptr);
        const double prediction_error =
            std::abs(reaction(problem, fitted, 2, -0.07, nullptr) - reference) / std::abs(reference);
        std::cout << "truth=" << truth.transpose() << ", fitted=" << fitted.transpose()
                  << ", parameter error=" << parameter_error << ", residual=" << r.norm()
                  << ", held-out error=" << prediction_error << ", forward solves=" << problem.solves << '\n';
        return result.status == solver::GaussNewtonStatus::Converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "volume_equilibrium_material_fit: " << error.what() << '\n';
        return 1;
    }
}
