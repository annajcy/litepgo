// Fit plane-stress KoiterStVK parameters from equilibrium boundary reactions.
// Build: cmake --build build/release --target shell_equilibrium_material_fit
// Run: build/release/example/fem/fitting/shell_equilibrium_material_fit

#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/fem/shell_fem_force_material_response.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include "litepgo/solver/static_equilibrium_response.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <iomanip>
#include <iostream>

// Equilibrium boundary-reaction calibration, with fixed thickness and a free interior vertex.
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
        using Force = fem::ShellFemForceMaterialResponse<fem::KoiterStVK>;
        struct Observation {
            Vec reaction, displacement;
            Eigen::MatrixXd reaction_J;
            SpMat state_J;
        };
        // Re-solve free DOFs at each trial; prescribed boundary values are parameter-independent.
        const auto experiment = [&](const Vec& theta, double load, bool derivatives) {
            const Vec initial = load * prescribed;
            solver::NewtonOptions options;
            options.gradient_tolerance = 1e-12;
            options.max_iterations = 100;
            // Partial force derivative at fixed displacement; equilibrium supplies du/dp separately.
            const auto partial = [&](const Vec& p, const Vec& u) -> SpMat {
                return Force(dof::ConstantDofMap(topology, 2), map, rest, u, thickness).eval_jacobian(p);
            };
            solver::StaticEquilibriumResponse equilibrium(
                2, fixed, initial, options,
                [&](const Vec& p) -> std::shared_ptr<energy::Energy> {
                    return std::make_shared<Elastic>(map, rest, thickness, std::vector<Vec>(map.num_elements(), p));
                },
                partial);
            Observation observation;
            observation.state_J = equilibrium.jacobian_pattern();
            observation.displacement = equilibrium.eval(theta, derivatives ? &observation.state_J : nullptr);
            const auto& u = observation.displacement;
            Elastic elastic(map, rest, thickness, std::vector<Vec>(map.num_elements(), theta));
            const Vec gradient = elastic.eval_gradient(u);
            if (fixed.reduce(gradient).lpNorm<Eigen::Infinity>() > 2e-12)
                throw std::runtime_error("Invalid equilibrium");
            observation.reaction.resize(fixed_ids.size());
            Eigen::MatrixXd total;
            if (derivatives) {
                total = Eigen::MatrixXd(partial(theta, u)) + elastic.eval_hessian(u) * observation.state_J;
                observation.reaction_J.resize(fixed_ids.size(), 2);
            }
            for (std::size_t i = 0; i < fixed_ids.size(); ++i) {
                const int r = fixed_ids[i];
                if (u[r] != initial[r])
                    throw std::runtime_error("Changed boundary values");
                observation.reaction[i] = gradient[r];
                if (derivatives)
                    observation.reaction_J.row(i) = total.row(r);
            }
            return observation;
        };
        const Eigen::Vector2d truth(2, 3), initial(.7, 1.2);
        const Vec target = experiment(truth, 1, false).reaction;
        const double scale = target.norm();
        const auto reference = experiment(initial, 1, true);
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(reference.reaction_J);
        const auto singular = svd.singularValues();
        const double singular_ratio = singular[1] / singular[0];
        if (!(scale > 0) || !(singular_ratio > 1e-6) || !(reference.state_J.norm() > 1e-5))
            throw std::runtime_error("Experiment must excite both parameters and equilibrium sensitivity");
        double fd_error = 0;
        for (double h : {1e-3, 1e-4})
            for (int c = 0; c < 2; ++c) {
                Eigen::Vector2d plus = initial, minus = initial;
                plus[c] += h;
                minus[c] -= h;
                const auto forward = experiment(plus, 1, false), backward = experiment(minus, 1, false);
                const Vec force_error = (forward.reaction - backward.reaction) / (2 * h) - reference.reaction_J.col(c);
                const Vec state_error =
                    (forward.displacement - backward.displacement) / (2 * h) - reference.state_J.col(c);
                fd_error =
                    std::max({fd_error, force_error.lpNorm<Eigen::Infinity>(), state_error.lpNorm<Eigen::Infinity>()});
            }
        if (fd_error > 2e-7)
            throw std::runtime_error("Equilibrium/reaction derivatives failed re-solve finite differences");
        class Residual final : public energy::FixedPatternResponse {
          public:
            Residual(const decltype(experiment)& experiment, const Vec& target, double scale)
                : FixedPatternResponse(Eigen::MatrixXd::Ones(target.size(), 2).sparseView()), experiment_(experiment),
                  target_(target), scale_(scale) {}

          private:
            Vec eval_values(const ConstVecRef& p, SpMat* J) const override {
                if (p[1] <= 0 || p.sum() <= 0)
                    throw std::domain_error("Require mu > 0 and lambda_ps + mu > 0");
                const auto observation = experiment_(p, 1, J != nullptr);
                if (J)
                    for (int c = 0; c < 2; ++c)
                        for (SpMat::InnerIterator it(*J, c); it; ++it)
                            it.valueRef() = observation.reaction_J(it.row(), c) / scale_;
                return (observation.reaction - target_) / scale_;
            }
            const decltype(experiment)& experiment_;
            const Vec& target_;
            double scale_;
        } residual(experiment, target, scale);
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-10;
        // Fit the measured response using its analytic parameter Jacobian.
        const auto result = solver::gauss_newton_solve(residual, initial, options);
        const double parameter_error = (result.x - truth).norm() / truth.norm();
        const double relative_residual = (experiment(result.x, 1, false).reaction - target).norm() / scale;
        const Vec held_out = experiment(truth, .6, false).reaction;
        const double held_out_error = (experiment(result.x, .6, false).reaction - held_out).norm() / held_out.norm();
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        std::cout << std::setprecision(12) << "fitted=" << result.x.transpose() << ", converged=" << converged
                  << ", parameter_error=" << parameter_error << ", relative_residual=" << relative_residual
                  << ", held_out_error=" << held_out_error << ", fd_error=" << fd_error
                  << ", singular_ratio=" << singular_ratio << '\n';
        return converged && parameter_error < 1e-6 && relative_residual < 1e-8 && held_out_error < 1e-8 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shell_equilibrium_material_fit: " << error.what() << '\n';
        return 1;
    }
}
