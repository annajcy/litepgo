// Recover shared positive diagonal shell Fp from equilibrium shapes and reactions.
// Build: cmake --build build/release --target shell_equilibrium_fp_fit
// Run: build/release/examples/fem/fitting/shell_equilibrium_fp_fit

#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/elementwise_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/parameter/log_diagonal_fp_response.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/fem/shell_fem_force_fp_response.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include "litepgo/solver/static_equilibrium_response.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

// Recover shared diagonal Fp from equilibrium shapes and boundary reactions.
int main() {
    using namespace litepgo;
    try {
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
        using Elastic = fem::ShellFemEnergy<fem::KoiterStVK>;
        using Force = fem::ShellFemForceFpResponse<fem::KoiterStVK>;
        const std::vector<Vec> theta(map.num_elements(), Eigen::Vector2d(2, 3));
        const Vec truth = Eigen::Vector2d(1.06, .95);
        constexpr int n = 2;
        const int num_free = fixed.num_free();
        if (num_free == 0 || fixed_ids.empty())
            throw std::runtime_error("Require both fixed and free DOFs");
        const dof::ElementDofMap physical_map(topology, 9);
        auto plastic = std::make_shared<energy::ElementwiseResponse>(dof::ConstantDofMap(topology, n),
                                                                     std::make_shared<fem::LogDiagonalFpResponse>(n));
        const auto make_energy = [&](const Vec& p) -> std::shared_ptr<energy::Energy> {
            const Vec packed = plastic->eval_value(p);
            std::vector<Eigen::Matrix3d> fields;
            for (int e = 0; e < map.num_elements(); ++e)
                fields.emplace_back(packed.segment(9 * e, 9).reshaped(3, 3));
            return std::make_shared<Elastic>(map, rest, thickness, theta, std::move(fields));
        };
        const auto make_response = [&](const Vec& u) {
            return std::make_shared<energy::ComposedResponse>(
                std::make_shared<Force>(physical_map, map, rest, u, thickness, theta), plastic);
        };
        const auto partial = [&](const Vec& p, const Vec& u) { return make_response(u)->eval_jacobian(p); };
        struct Observation {
            Vec value;
            Eigen::MatrixXd jacobian;
        };
        // Re-solve free DOFs at each trial; prescribed boundary values are parameter-independent.
        const auto experiment = [&](const Vec& p, double load, bool derivatives) {
            const Vec initial = load * prescribed;
            solver::NewtonOptions options;
            options.gradient_tolerance = 1e-13;
            options.max_iterations = 100;
            solver::StaticEquilibriumResponse equilibrium(n, fixed, initial, options, make_energy, make_response);
            SpMat J = equilibrium.jacobian_pattern();
            const Vec u = equilibrium.eval(p, derivatives ? &J : nullptr);
            const auto elastic = make_energy(p);
            const Vec gradient = elastic->eval_gradient(u);
            if (fixed.reduce(gradient).lpNorm<Eigen::Infinity>() > 2e-11)
                throw std::runtime_error("Invalid equilibrium");
            Observation observation;
            observation.value.resize(map.num_dofs());
            observation.value.head(num_free) = fixed.reduce(u);
            Eigen::MatrixXd total;
            if (derivatives) {
                if (J.norm() < 1e-5)
                    throw std::runtime_error("Experiment must have nonzero equilibrium sensitivity");
                total = Eigen::MatrixXd(partial(p, u)) + elastic->eval_hessian(u) * J;
                observation.jacobian.resize(map.num_dofs(), n);
                for (int c = 0; c < n; ++c)
                    observation.jacobian.col(c).head(num_free) = fixed.reduce(Vec(J.col(c)));
            }
            for (std::size_t i = 0; i < fixed_ids.size(); ++i) {
                const int r = fixed_ids[i];
                if (u[r] != initial[r])
                    throw std::runtime_error("Changed boundary values");
                observation.value[num_free + i] = gradient[r];
                if (derivatives)
                    observation.jacobian.row(num_free + i) = total.row(r);
            }
            return observation;
        };
        const Vec truth_p = truth.array().log();
        const Vec target = experiment(truth_p, 1, false).value;
        Vec scales(map.num_dofs());
        scales.head(num_free).setConstant(std::max(target.head(num_free).norm(), 1e-8));
        scales.tail(fixed_ids.size()).setConstant(std::max(target.tail(fixed_ids.size()).norm(), 1e-8));
        class Residual final : public energy::FixedPatternResponse {
          public:
            Residual(const decltype(experiment)& experiment, const Vec& target, const Vec& scales, int parameters)
                : FixedPatternResponse(Eigen::MatrixXd::Ones(target.size(), parameters).sparseView()),
                  experiment_(experiment), target_(target), scales_(scales) {}

          private:
            Vec eval_values(const ConstVecRef& p, SpMat* J) const override {
                const auto observation = experiment_(p, 1, J != nullptr);
                if (J)
                    for (int c = 0; c < num_inputs(); ++c)
                        for (SpMat::InnerIterator it(*J, c); it; ++it)
                            it.valueRef() = observation.jacobian(it.row(), c) / scales_[it.row()];
                return (observation.value - target_).cwiseQuotient(scales_);
            }
            const decltype(experiment)& experiment_;
            const Vec& target_;
            const Vec& scales_;
        } residual(experiment, target, scales, n);
        const auto check_rank = [&](const Vec& p) {
            const Eigen::MatrixXd J = residual.eval_jacobian(p);
            const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
            const auto singular = svd.singularValues();
            const double ratio = singular[n - 1] / singular[0];
            if (!(ratio > 1e-6))
                throw std::runtime_error("Observations cannot identify every plastic stretch");
            return ratio;
        };
        const Vec initial = Vec::Zero(n);
        const double initial_ratio = check_rank(initial), truth_ratio = check_rank(truth_p);
        double state_fd_error = 0, reaction_fd_error = 0;
        for (const Vec& p : {initial, truth_p}) {
            const auto reference = experiment(p, 1, true);
            for (double h : {3e-5, 1e-5})
                for (int c = 0; c < n; ++c) {
                    Vec plus = p, minus = p;
                    plus[c] += h;
                    minus[c] -= h;
                    const Vec forward = experiment(plus, 1, false).value, backward = experiment(minus, 1, false).value;
                    const Vec error =
                        ((forward - backward) / (2 * h) - reference.jacobian.col(c)).cwiseQuotient(scales);
                    state_fd_error = std::max(state_fd_error, error.head(num_free).lpNorm<Eigen::Infinity>());
                    reaction_fd_error =
                        std::max(reaction_fd_error, error.tail(fixed_ids.size()).lpNorm<Eigen::Infinity>());
                }
        }
        if (std::max(state_fd_error, reaction_fd_error) > 2e-6)
            throw std::runtime_error("Re-solve FD failed: state=" + std::to_string(state_fd_error) +
                                     ", reaction=" + std::to_string(reaction_fd_error));
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-10;
        // Fit the measured response using its analytic parameter Jacobian.
        const auto result = solver::gauss_newton_solve(residual, initial, options);
        const Vec fitted = result.x.array().exp();
        const double fitted_ratio = check_rank(result.x);
        const double parameter_error = (fitted - truth).norm() / truth.norm();
        const double relative_residual = residual.eval_value(result.x).norm();
        const Vec held_out = experiment(truth_p, .6, false).value;
        const double held_out_error = (experiment(result.x, .6, false).value - held_out).cwiseQuotient(scales).norm();
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        std::cout << std::setprecision(12) << "truth=" << truth.transpose() << ", fitted=" << fitted.transpose()
                  << ", converged=" << converged << ", parameter_error=" << parameter_error
                  << ", relative_residual=" << relative_residual << ", held_out_error=" << held_out_error
                  << ", state_fd_error=" << state_fd_error << ", reaction_fd_error=" << reaction_fd_error
                  << ", initial_ratio=" << initial_ratio << ", truth_ratio=" << truth_ratio
                  << ", fitted_ratio=" << fitted_ratio << '\n';
        return converged && parameter_error < 1e-6 && relative_residual < 1e-8 && held_out_error < 1e-8 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shell_equilibrium_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
