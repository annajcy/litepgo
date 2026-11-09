#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>
// Fit three shared log plastic stretches from equilibrium displacement and reaction.
// Geometry, material theta=(2,3), and boundary displacements are fixed.
// Run: build/release/examples/fem/fitting/volume_equilibrium_fp_fit
#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/elementwise_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "litepgo/fem/assembly/volumetric_fem_force_fp_response.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/response/log_diagonal_fp_response.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include "litepgo/solver/static_equilibrium_response.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>

namespace {
using namespace litepgo;

using ElasticEnergy = fem::VolumetricFemEnergy<fem::TetP1Basis, fem::TetCentroidQuadrature>;
using ForceFpResponse = fem::VolumetricFemForceFpResponse<fem::TetP1Basis, fem::TetCentroidQuadrature>;

struct Problem {
    dof::VertexDofMap displacement_map;
    std::vector<std::shared_ptr<litepgo::energy::Response>> geometry;
    std::vector<std::shared_ptr<litepgo::energy::Response>> theta;
    dof::DofReduction fixed;
    Vec prescribed;
    dof::ConstantDofMap parameter_map;
    dof::ElementDofMap physical_map;
};

Problem make_problem() {
    const auto volume = std::get<mesh::TetMesh>(mesh::read_veg(LITEPGO_ASSETS_DIR "/box-tet.veg"));
    const auto& topology = volume.topology;
    const dof::VertexDofMap displacement_map(topology, 3);
    std::vector<std::shared_ptr<litepgo::energy::Response>> geometry;
    for (int e = 0; e < displacement_map.num_elements(); ++e)
        geometry.push_back(
            std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(displacement_map.gather(e, volume.positions)));
    std::vector<std::shared_ptr<litepgo::energy::Response>> theta(
        displacement_map.num_elements(),
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(2, 3)));

    // Clamp the bottom; prescribe shear and extension on the top.
    const auto nodes = volume.positions.reshaped<Eigen::RowMajor>(topology.num_vertices(), 3);
    const double bottom = nodes.col(2).minCoeff(), top = nodes.col(2).maxCoeff(), height = top - bottom;
    if (!(height > 0))
        throw std::runtime_error("Require nonempty volume with positive height");
    std::vector<int> fixed_ids;
    Vec prescribed = Vec::Zero(displacement_map.num_dofs());
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
    dof::DofReduction fixed(displacement_map.num_dofs(), fixed_ids);
    if (fixed.num_free() == 0 || fixed_ids.empty())
        throw std::runtime_error("Require both fixed and free DOFs");
    return {displacement_map,
            std::move(geometry),
            std::move(theta),
            std::move(fixed),
            std::move(prescribed),
            dof::ConstantDofMap(topology, 3),
            dof::ElementDofMap(topology, 9)};
}

struct Factories {
    solver::StaticEquilibriumResponse::MakeEnergy make_energy;
    solver::StaticEquilibriumResponse::MakeResponse make_response;
};

// The problem outlives both factories; they share the same p -> per-element Fp transform.
Factories make_factories(const Problem& problem) {
    auto plastic = std::make_shared<energy::ElementwiseResponse>(problem.parameter_map,
                                                                 std::make_shared<fem::LogDiagonalFpResponse>(3));
    auto make_energy = [&problem, plastic](const Vec& p) -> std::shared_ptr<energy::Energy> {
        const Vec packed = plastic->eval_value(p);
        std::vector<std::shared_ptr<litepgo::energy::Response>> fields;
        for (int e = 0; e < problem.displacement_map.num_elements(); ++e)
            fields.push_back(std::make_shared<litepgo::energy::AffineResponse>(
                litepgo::SpMat((packed.segment(9 * e, 9)).size(), 3), packed.segment(9 * e, 9)));
        return std::make_shared<ElasticEnergy>(
            std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<fem::NeoHookean>>(), problem.displacement_map,
            problem.geometry, std::move(fields), problem.theta);
    };
    auto make_response = [&problem, plastic](const Vec& u) -> std::shared_ptr<energy::Response> {
        auto force = std::make_shared<ForceFpResponse>(
            std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<fem::NeoHookean>>(), problem.physical_map,
            problem.displacement_map, problem.geometry, u, problem.theta);
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
    options.gradient_tolerance = 1e-11;
    options.max_iterations = 100;
    solver::StaticEquilibriumResponse equilibrium(3, problem.fixed, load * problem.prescribed, options,
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
        observation.jacobian.resize(displacement.size(), 3);
        for (int c = 0; c < 3; ++c)
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
        : FixedPatternResponse(Eigen::MatrixXd::Ones(target.size(), 3).sparseView()),
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
        const Vec truth = Eigen::Vector3d(1.04, .97, 1.02);
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
        const auto result = solver::gauss_newton_solve(residual, Vec::Zero(3), options);
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
        std::cerr << "volume_equilibrium_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
