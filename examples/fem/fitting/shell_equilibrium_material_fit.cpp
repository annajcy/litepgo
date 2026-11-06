#include <memory>
// Fit two material parameters from equilibrium reactions.
// Geometry, thickness, and boundary displacements are fixed.
// Run: build/release/examples/fem/fitting/shell_equilibrium_material_fit
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/fem/shell_fem_force_material_response.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include "litepgo/solver/static_equilibrium_response.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>

namespace {
using namespace litepgo;
using ElasticEnergy = fem::ShellFemEnergy<fem::KoiterStVK>;
using ForceParameterResponse = fem::ShellFemForceMaterialResponse<fem::KoiterStVK>;
struct Problem {
    dof::StencilDofMap displacement_map;
    std::vector<fem::TriangleMidEdgeRestData> rest;
    std::vector<double> thickness;
    dof::DofReduction fixed;
    Vec prescribed;
    dof::ConstantDofMap parameter_map;
};
Problem make_problem() {
    Eigen::Matrix<double, 9, 3> nodes;
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
            nodes.row(3 * y + x) = Eigen::Vector3d(x - 1, y - 1, 0);
    std::vector<mesh::TriangleTopology::Cell> faces;
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            const int a = 3 * y + x;
            faces.push_back({a, a + 1, a + 4});
            faces.push_back({a, a + 4, a + 3});
        }
    const mesh::TriangleTopology topology(9, faces);
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
    const dof::StencilDofMap map(9, stencils.vertices, 3);
    const Vec positions = nodes.reshaped<Eigen::RowMajor>();
    std::vector<fem::TriangleMidEdgeRestData> rest;
    for (int e = 0; e < map.num_elements(); ++e)
        rest.emplace_back(map.gather(e, positions), stencils.opposite[e]);
    const std::vector<double> thickness(map.num_elements(), .1);
    std::vector<int> fixed_ids;
    for (int i = 0; i < 27; ++i)
        if (i / 3 != 4)
            fixed_ids.push_back(i);
    const dof::DofReduction fixed(27, fixed_ids);
    Vec prescribed(27);
    for (int v = 0; v < 9; ++v) {
        const double X = nodes(v, 0), Y = nodes(v, 1);
        prescribed.segment<3>(3 * v) =
            Eigen::Vector3d(.08 * X + .025 * X * X + .04 * Y, .03 * Y + .02 * X * Y, .08 * X * Y + .03 * X * X);
    }
    return {map, rest, thickness, fixed, prescribed, dof::ConstantDofMap(topology, 2)};
}

struct Factories {
    solver::StaticEquilibriumResponse::MakeEnergy make_energy;
    solver::StaticEquilibriumResponse::MakeResponse make_response;
};

// The problem outlives both factories; both describe the same material parameters.
Factories make_factories(const Problem& problem) {
    auto make_energy = [&problem](const Vec& theta) -> std::shared_ptr<energy::Energy> {
        return std::make_shared<ElasticEnergy>(problem.displacement_map, problem.rest, problem.thickness,
                                               std::vector<Vec>(problem.displacement_map.num_elements(), theta));
    };
    auto make_response = [&problem](const Vec& u) -> std::shared_ptr<energy::Response> {
        return std::make_shared<ForceParameterResponse>(problem.parameter_map, problem.displacement_map, problem.rest,
                                                        u, problem.thickness);
    };
    return {std::move(make_energy), std::move(make_response)};
}

struct Observation {
    Vec value;
    Eigen::MatrixXd jacobian;
};

Observation measure(const Problem& problem, const Factories& factories, const Vec& p, double load = 1.,
                    bool need_jacobian = false) {
    // 1. Solve equilibrium, starting from the prescribed displacement every time.
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-12;
    options.max_iterations = 100;
    solver::StaticEquilibriumResponse equilibrium(2, problem.fixed, load * problem.prescribed, options,
                                                  factories.make_energy, factories.make_response);
    SpMat state_jacobian = SpMat();
    const Vec displacement = (need_jacobian ? ([&] {
        auto result = equilibrium.eval_value_and_jacobian(p);
        state_jacobian = std::move(result.second);
        return std::move(result.first);
    }())
                                            : equilibrium.eval_value(p));

    // 2. Compute force and its total derivative along the equilibrium branch.
    const auto elastic = factories.make_energy(p);
    const Vec force = elastic->eval_gradient(displacement);
    Eigen::MatrixXd reaction_jacobian;
    if (need_jacobian) {
        const SpMat force_parameter_jacobian = factories.make_response(displacement)->eval_jacobian(p);
        const SpMat stiffness = elastic->eval_hessian(displacement);
        reaction_jacobian = Eigen::MatrixXd(force_parameter_jacobian) + stiffness * state_jacobian;
    }

    // 3. Observe reactions at fixed DOFs.
    const auto& fixed_ids = problem.fixed.fixed_dofs();
    Observation observation;
    observation.value.resize(fixed_ids.size());
    if (need_jacobian)
        observation.jacobian.resize(fixed_ids.size(), 2);
    for (std::size_t i = 0; i < fixed_ids.size(); ++i) {
        observation.value[i] = force[fixed_ids[i]];
        if (need_jacobian)
            observation.jacobian.row(i) = reaction_jacobian.row(fixed_ids[i]);
    }
    return observation;
}

class Residual final : public energy::FixedPatternResponse {
  public:
    using Measurement = std::function<Observation(const Vec&, bool)>;
    Residual(Measurement measurement, Vec target, Vec scales)
        : FixedPatternResponse(Eigen::MatrixXd::Ones(target.size(), 2).sparseView()),
          measurement_(std::move(measurement)), target_(std::move(target)), scales_(std::move(scales)) {}

  private:
    Vec eval_impl(const ConstVecRef& p, SpMat* J) override {
        if (p[1] <= 0 || p.sum() <= 0)
            throw std::domain_error("Invalid material parameters");
        const auto observation = measurement_(p, J != nullptr);
        if (J)
            for (int c = 0; c < num_inputs(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = observation.jacobian(it.row(), c) / scales_[it.row()];
        return (observation.value - target_).cwiseQuotient(scales_);
    }
    Measurement measurement_;
    Vec target_, scales_;
};
} // namespace

int main() {
    try {
        const auto problem = make_problem();
        const auto factories = make_factories(problem);
        const auto measurement = [&](const Vec& p, bool derivatives) {
            return measure(problem, factories, p, 1., derivatives);
        };

        // Synthetic target; each observation block uses a fixed normalization.
        const Vec truth = Eigen::Vector2d(2, 3);
        const Vec truth_p = truth;
        const Vec target = measurement(truth_p, false).value;
        const double scale = target.norm();
        if (!(scale > 0))
            throw std::runtime_error("Calibration requires nonzero target reactions");
        const Vec scales = Vec::Constant(target.size(), scale);
        Residual residual(measurement, target, scales);

        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-10;
        const auto result = solver::gauss_newton_solve(residual, Eigen::Vector2d(.7, 1.2), options);
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        const Vec fitted = result.x;
        const double parameter_error = (fitted - truth).norm() / truth.norm();
        const double relative_residual = residual.eval_value(result.x).norm();

        // Predict a load that was not used for fitting.
        const Vec held_out = measure(problem, factories, truth_p, .6).value;
        const Vec prediction = measure(problem, factories, result.x, .6).value;
        const double held_out_error = (prediction - held_out).norm() / held_out.norm();
        std::cout << std::setprecision(12) << "fitted=" << fitted.transpose() << ", converged=" << converged
                  << ", parameter_error=" << parameter_error << ", relative_residual=" << relative_residual
                  << ", held_out_error=" << held_out_error << '\n';
        return converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shell_equilibrium_material_fit: " << error.what() << '\n';
        return 1;
    }
}
