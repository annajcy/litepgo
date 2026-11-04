#include <functional>
// Recover shared positive diagonal volume Fp from equilibrium shapes and reactions.
// Build: cmake --build build/release --target volume_equilibrium_fp_fit
// Run: build/release/examples/fem/fitting/volume_equilibrium_fp_fit

#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_fp_response.hpp"
#include "litepgo/mesh/io/veg.hpp"
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
        const auto volume = std::get<mesh::TetMesh>(mesh::read_veg(LITEPGO_ASSETS_DIR "/box-tet.veg"));
        const auto& topology = volume.topology;
        const dof::VertexDofMap map(topology, 3);
        using XField = fem::InterpolatedField3D<fem::TetP1Basis>;
        using Field = fem::IdentityField3D;
        std::vector<XField> geometry;
        for (int e = 0; e < map.num_elements(); ++e)
            geometry.emplace_back(map.gather(e, volume.positions));
        const auto nodes = volume.positions.reshaped<Eigen::RowMajor>(topology.num_vertices(), 3);
        const double bottom = nodes.col(2).minCoeff(), top = nodes.col(2).maxCoeff(), height = top - bottom;
        if (!(height > 0))
            throw std::runtime_error("Require nonempty volume with positive height");
        std::vector<int> fixed_ids;
        Vec prescribed = Vec::Zero(map.num_dofs());
        for (int v = 0; v < topology.num_vertices(); ++v) {
            const bool upper = std::abs(nodes(v, 2) - top) < height * 1e-8;
            if (upper || std::abs(nodes(v, 2) - bottom) < height * 1e-8)
                for (int c = 0; c < 3; ++c)
                    fixed_ids.push_back(3 * v + c);
            if (upper) {
                prescribed[3 * v] = .04 * height;
                prescribed[3 * v + 2] = .05 * height;
            }
        }
        const dof::DofReduction fixed(map.num_dofs(), fixed_ids);
        using Elastic = fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature, XField,
                                                 Field, Field>;
        using Force = fem::VolumetricFemForceFpResponse<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature,
                                                        XField, Field>;
        const std::vector<Field> theta(map.num_elements(), Field(Eigen::Vector2d(2, 3)));
        const Vec truth = Eigen::Vector3d(1.04, .97, 1.02);
        constexpr int n = 3;
        const int num_free = fixed.num_free();
        if (num_free == 0 || fixed_ids.empty())
            throw std::runtime_error("Require both fixed and free DOFs");
        const auto plastic = [&](const Vec& p) -> Eigen::Matrix3d {
            const Vec stretches = p.array().exp();
            if (!stretches.allFinite() || (stretches.array() <= 0).any())
                throw std::domain_error("Unrepresentable plastic stretches");
            Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity();
            Fp.diagonal().head(n) = stretches;
            return Fp;
        };
        const auto make_energy = [&](const Vec& p) -> std::shared_ptr<energy::Energy> {
            const Eigen::Matrix3d Fp = plastic(p);
            return std::make_shared<Elastic>(map, geometry,
                                             std::vector<Field>(map.num_elements(), Field(Vec(Fp.reshaped()))), theta);
        };
        // Partial force derivative at fixed displacement; equilibrium supplies du/dp separately.
        const auto partial = [&](const Vec& p, const Vec& u) -> SpMat {
            const Eigen::Matrix3d Fp = plastic(p);
            const Vec packed = Fp.reshaped();
            const SpMat full = Force(dof::ConstantDofMap(topology, 9), map, geometry, u, theta).eval_jacobian(packed);
            Eigen::MatrixXd B(map.num_dofs(), n);
            // Column-major diagonal slots; chain rule through exp(p).
            for (int c = 0; c < n; ++c)
                B.col(c) = full.col(4 * c) * Fp(c, c);
            return B.sparseView();
        };
        struct Observation {
            Vec value;
            Eigen::MatrixXd jacobian;
        };
        // Re-solve free DOFs at each trial; prescribed boundary values are parameter-independent.
        const auto experiment = [&](const Vec& p, double load, bool derivatives) {
            const Vec initial = load * prescribed;
            solver::NewtonOptions options;
            options.gradient_tolerance = 1e-11;
            options.max_iterations = 100;
            // One analytic model supplies both state and parameter derivatives.
            class PlasticModel final : public energy::ParametricEnergy {
                using Factory = std::function<std::shared_ptr<energy::Energy>(const Vec&)>;
                using Partial = std::function<SpMat(const Vec&, const Vec&)>;

              public:
                PlasticModel(Factory factory, Partial partial, const Vec& sample)
                    : ParametricEnergy(factory(sample)->hessian_pattern(),
                                       Eigen::MatrixXd::Ones(factory(sample)->num_dofs(), sample.size()).sparseView()),
                      factory_(std::move(factory)), partial_(std::move(partial)) {}

              private:
                double eval_state_values(const ConstVecRef& u, const ConstVecRef& p, Vec* g, SpMat* K,
                                         energy::HessianKind kind) const override {
                    auto potential = factory_(p);
                    potential->update_pattern(u);
                    return potential->eval(u, g, K, kind);
                }
                void eval_parameter_jacobian(const ConstVecRef& u, const ConstVecRef& p, SpMat& B) const override {
                    const SpMat source = partial_(p, u);
                    for (int c = 0; c < B.outerSize(); ++c)
                        for (SpMat::InnerIterator it(B, c); it; ++it)
                            it.valueRef() = source.coeff(it.row(), c);
                }
                double eval_feasible_step(const ConstVecRef& u, const ConstVecRef& p,
                                          const ConstVecRef& du) const override {
                    return factory_(p)->max_feasible_step(u, du);
                }
                Factory factory_;
                Partial partial_;
            };
            auto model = std::make_shared<PlasticModel>(make_energy, partial, p);
            solver::StaticEquilibriumResponse equilibrium(model, fixed, initial, options);
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
            for (double h : {1e-4, 1e-5})
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
        std::cerr << "volume_equilibrium_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
