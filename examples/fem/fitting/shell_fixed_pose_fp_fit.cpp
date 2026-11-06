#include <memory>
// Fit two shared positive in-plane Fp stretches at a prescribed shell pose.
// Build: cmake --build build/release --target shell_fixed_pose_fp_fit
// Run: build/release/examples/fem/fitting/shell_fixed_pose_fp_fit

#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/fem/shell_fem_force_fp_response.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/gauss_newton.hpp"

#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/fem/parameter/log_diagonal_fp_response.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Fit two shared in-plane plastic stretches at a prescribed pose.
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
    const auto mesh = mesh::read_obj(LITEPGO_ASSETS_DIR "/shell.obj");
    const auto& topology = mesh.topology;
    if (topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty sheet");
    const auto nodes = mesh.positions.reshaped<Eigen::RowMajor>(topology.num_vertices(), 3);
    const Eigen::Vector3d lower = nodes.colwise().minCoeff().transpose();
    const Eigen::Vector3d upper = nodes.colwise().maxCoeff().transpose();
    const Eigen::Vector3d center = (lower + upper) / 2;
    const double size = (upper - lower).head<2>().maxCoeff(), tolerance = size * 1e-8;
    if (!((upper - lower).head<2>().array() > tolerance).all() || upper.z() - lower.z() > tolerance)
        throw std::runtime_error("Example requires a flat sheet in the xy plane");
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
    const dof::StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    std::vector<fem::TriangleMidEdgeRestData> rest;
    for (int e = 0; e < map.num_elements(); ++e)
        rest.emplace_back(map.gather(e, mesh.positions), stencils.opposite[e]);
    const std::vector<double> thickness(map.num_elements(), 0.1);

    // Anisotropic stretch, shear and saddle bending excite both in-plane plastic stretches.
    Vec u(map.num_dofs());
    for (int v = 0; v < topology.num_vertices(); ++v) {
        const Eigen::Vector3d X = nodes.row(v).transpose() - center;
        u.segment<3>(3 * v) = Eigen::Vector3d(.08 * X.x() + .04 * X.y(), -.03 * X.y(), .15 * X.x() * X.y() / size);
    }
    const Eigen::Vector2d truth(1.06, .95), initial = Eigen::Vector2d::Ones();
    const Eigen::Matrix3d true_Fp = Eigen::Vector3d(truth[0], truth[1], 1).asDiagonal();
    const std::vector<Vec> theta(map.num_elements(), Eigen::Vector2d(2, 3));
    // The energy fixes Fp, while the force response below makes it the active input.
    fem::ShellFemEnergy<fem::KoiterStVK> target_energy(map, rest, thickness, theta,
                                                       std::vector<Eigen::Matrix3d>(map.num_elements(), true_Fp));
    const Vec target = target_energy.eval_gradient(u);
    const double scale = target.norm();
    if (!(scale > 0) || !std::isfinite(scale))
        throw std::runtime_error("Calibration requires finite nonzero target forces");
    auto force = std::make_shared<fem::ShellFemForceFpResponse<fem::KoiterStVK>>(dof::ConstantDofMap(topology, 9), map,
                                                                                 rest, u, thickness, theta);

    auto parameterized =
        std::make_shared<energy::ComposedResponse>(force, std::make_shared<fem::LogDiagonalFpResponse>(2));
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
        const Eigen::Vector2d fitted = result.x.array().exp();
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
        std::cerr << "shell_fixed_pose_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
