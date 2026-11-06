#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>
// Fit three shared positive diagonal Fp stretches at a prescribed tetrahedral pose.
// Build: cmake --build build/release --target volume_fixed_pose_fp_fit
// Run: build/release/examples/fem/fitting/volume_fixed_pose_fp_fit

#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_fp_response.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/gauss_newton.hpp"

#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/fem/parameter/log_diagonal_fp_response.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Fit three shared plastic stretches at a prescribed pose; no equilibrium or plastic evolution solve.
namespace {
using namespace litepgo;
class Residual final : public energy::FixedPatternResponse {
  public:
    Residual(std::shared_ptr<energy::Response> force, Vec target, double scale)
        : FixedPatternResponse(Eigen::MatrixXd::Ones(force->num_outputs(), force->num_inputs()).sparseView()),
          force_(std::move(force)), target_(std::move(target)), scale_(scale) {}

  private:
    Vec eval_impl(const ConstVecRef& p, SpMat* J) override {
        Vec value;
        if (J) {
            auto evaluated = force_->eval_value_and_jacobian(p);
            value = std::move(evaluated.first);
            const SpMat& derivative = evaluated.second;
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = derivative.coeff(it.row(), c);
        } else {
            value = force_->eval_value(p);
        }
        if (J)
            *J /= scale_;
        return (value - target_) / scale_;
    }
    std::shared_ptr<energy::Response> force_;
    Vec target_;
    double scale_;
};
struct Problem {
    std::shared_ptr<energy::Response> residual;
    Vec truth, initial;
    double scale;
};
Problem make_problem() {
    const auto mesh = std::get<mesh::TetMesh>(mesh::read_veg(LITEPGO_ASSETS_DIR "/box-tet.veg"));
    if (mesh.topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty tetrahedral mesh");
    const dof::VertexDofMap map(mesh.topology, 3);

    std::vector<std::shared_ptr<litepgo::energy::Response>> geometry;
    for (int e = 0; e < map.num_elements(); ++e)
        geometry.push_back(std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(map.gather(e, mesh.positions)));
    const auto nodes = mesh.positions.reshaped<Eigen::RowMajor>(mesh.topology.num_vertices(), 3);
    const Eigen::Vector3d center = nodes.colwise().mean().transpose();
    Eigen::Matrix3d F;
    F << 1.12, .08, .02, .01, .96, .06, .03, 0, 1.07;
    Vec u(map.num_dofs());
    for (int v = 0; v < mesh.topology.num_vertices(); ++v)
        u.segment<3>(3 * v) = (F - Eigen::Matrix3d::Identity()) * (nodes.row(v).transpose() - center);

    const Eigen::Vector3d truth(1.08, .94, 1.03), initial = Eigen::Vector3d::Ones();
    const Eigen::Matrix3d true_Fp = truth.asDiagonal();
    const std::vector<std::shared_ptr<litepgo::energy::Response>> theta(
        geometry.size(),
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(2, 3)));
    using Elastic = fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature>;
    const Elastic elastic(map, geometry,
                          std::vector<std::shared_ptr<litepgo::energy::Response>>(
                              geometry.size(), std::make_shared<litepgo::energy::AffineResponse>(
                                                   litepgo::SpMat((true_Fp.reshaped()).size(), 3), true_Fp.reshaped())),
                          theta);
    // Target comes from the energy gradient, independently of the force-response path.
    const Vec target = elastic.eval_gradient(u);
    const double scale = target.norm();
    if (!(scale > 0) || !std::isfinite(scale))
        throw std::runtime_error("Calibration requires finite nonzero target forces");
    using Force = fem::VolumetricFemForceFpResponse<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature>;
    auto force = std::make_shared<Force>(dof::ConstantDofMap(mesh.topology, 9), map, geometry, u, theta);

    auto parameterized =
        std::make_shared<energy::ComposedResponse>(force, std::make_shared<fem::LogDiagonalFpResponse>(3));
    return {std::make_shared<Residual>(parameterized, target, scale), truth, initial, scale};
}
} // namespace
int main() {
    using namespace litepgo;
    try {
        const auto problem = make_problem();
        auto& residual = *problem.residual;
        const auto& truth = problem.truth;
        const auto& initial = problem.initial;
        const double scale = problem.scale;
        const Vec initial_p = initial.array().log();
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-12;
        // Fit the measured response using its analytic parameter Jacobian.
        const auto result = solver::gauss_newton_solve(residual, initial_p, options);
        const Eigen::Vector3d fitted = result.x.array().exp();
        const double parameter_error = (fitted - truth).norm() / truth.norm();
        const double relative_residual = residual.eval_value(result.x).norm();
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        std::cout << std::setprecision(12) << "truth=" << truth.transpose() << ", initial=" << initial.transpose()
                  << ", fitted=" << fitted.transpose() << ", converged=" << converged
                  << ", iterations=" << result.iterations << ", parameter_error=" << parameter_error
                  << ", force_residual=" << scale * relative_residual << ", relative_residual=" << relative_residual
                  << '\n';
        return converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "volume_fixed_pose_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
