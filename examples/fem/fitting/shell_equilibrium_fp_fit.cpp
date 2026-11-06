#include <memory>
// Fit two shared in-plane log plastic stretches from equilibrium displacement and reaction.
// Geometry, material theta=(2,3), and boundary displacements are fixed.
// Run: build/release/examples/fem/fitting/shell_equilibrium_fp_fit
#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/elementwise_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/response/log_diagonal_fp_response.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/assembly/shell_fem_force_fp_response.hpp"
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
using ForceFpResponse = fem::ShellFemForceFpResponse<fem::KoiterStVK>;
struct Problem {
    dof::StencilDofMap displacement_map;
    std::vector<fem::TriangleMidEdgeRestData> rest;
    std::vector<double> thickness;
    std::vector<Vec> theta;
    dof::DofReduction fixed;
    Vec prescribed;
    dof::ConstantDofMap parameter_map;
    dof::ElementDofMap physical_map;
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
    return {map,
            rest,
            thickness,
            std::vector<Vec>(map.num_elements(), Eigen::Vector2d(2, 3)),
            fixed,
            prescribed,
            dof::ConstantDofMap(topology, 2),
            dof::ElementDofMap(topology, 9)};
}

struct Factories {
    solver::StaticEquilibriumResponse::MakeEnergy make_energy;
    solver::StaticEquilibriumResponse::MakeResponse make_response;
};

// The problem outlives both factories; they share the same p -> per-element Fp transform.
Factories make_factories(const Problem& problem) {
    auto plastic = std::make_shared<energy::ElementwiseResponse>(problem.parameter_map,
                                                                 std::make_shared<fem::LogDiagonalFpResponse>(2));
    auto make_energy = [&problem, plastic](const Vec& p) -> std::shared_ptr<energy::Energy> {
        const Vec packed = plastic->eval_value(p);
        std::vector<Eigen::Matrix3d> fields;
        for (int e = 0; e < problem.displacement_map.num_elements(); ++e)
            fields.emplace_back(packed.segment(9 * e, 9).reshaped(3, 3));
        return std::make_shared<ElasticEnergy>(problem.displacement_map, problem.rest, problem.thickness, problem.theta,
                                               std::move(fields));
    };
    auto make_response = [&problem, plastic](const Vec& u) -> std::shared_ptr<energy::Response> {
        auto force = std::make_shared<ForceFpResponse>(problem.physical_map, problem.displacement_map, problem.rest, u,
                                                       problem.thickness, problem.theta);
        return std::make_shared<energy::ComposedResponse>(force, plastic);
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
    options.gradient_tolerance = 1e-13;
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

    // 3. Observe free displacement, followed by the reactions at fixed DOFs.
    const int num_free = problem.fixed.num_free();
    Observation observation;
    observation.value.resize(displacement.size());
    observation.value.head(num_free) = problem.fixed.reduce(displacement);
    if (need_jacobian) {
        observation.jacobian.resize(displacement.size(), 2);
        for (int c = 0; c < 2; ++c)
            observation.jacobian.col(c).head(num_free) = problem.fixed.reduce(Vec(state_jacobian.col(c)));
    }
    const auto& fixed_ids = problem.fixed.fixed_dofs();
    for (std::size_t i = 0; i < fixed_ids.size(); ++i) {
        observation.value[num_free + i] = force[fixed_ids[i]];
        if (need_jacobian)
            observation.jacobian.row(num_free + i) = reaction_jacobian.row(fixed_ids[i]);
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
        const Vec truth = Eigen::Vector2d(1.06, .95);
        const Vec truth_p = truth.array().log();
        const Vec target = measurement(truth_p, false).value;
        const int num_free = problem.fixed.num_free();
        const auto num_fixed = problem.fixed.fixed_dofs().size();
        Vec scales(target.size());
        scales.head(num_free).setConstant(std::max(target.head(num_free).norm(), 1e-8));
        scales.tail(num_fixed).setConstant(std::max(target.tail(num_fixed).norm(), 1e-8));
        Residual residual(measurement, target, scales);

        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-10;
        const auto result = solver::gauss_newton_solve(residual, Vec::Zero(2), options);
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        const Vec fitted = result.x.array().exp();
        const double parameter_error = (fitted - truth).norm() / truth.norm();
        const double relative_residual = residual.eval_value(result.x).norm();

        // Predict a load that was not used for fitting.
        const Vec held_out = measure(problem, factories, truth_p, .6).value;
        const Vec prediction = measure(problem, factories, result.x, .6).value;
        const double held_out_error = (prediction - held_out).cwiseQuotient(scales).norm();
        std::cout << std::setprecision(12) << "truth=" << truth.transpose() << ", fitted=" << fitted.transpose()
                  << ", converged=" << converged << ", parameter_error=" << parameter_error
                  << ", relative_residual=" << relative_residual << ", held_out_error=" << held_out_error << '\n';
        return converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shell_equilibrium_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
