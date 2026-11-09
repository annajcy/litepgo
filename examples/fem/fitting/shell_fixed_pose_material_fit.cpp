#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include <memory>
// Recover shared KoiterStVK parameters from forces at a prescribed shell pose.
// Build: cmake --build build/release --target shell_fixed_pose_material_fit
// Run: build/release/examples/fem/fitting/shell_fixed_pose_material_fit

#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/assembly/shell_fem_force_material_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/gauss_newton.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Fixed-pose force calibration: no equilibrium solve or thickness optimization.
namespace {
using namespace litepgo;
class MaterialResidual final : public energy::FixedPatternResponse {
  public:
    MaterialResidual(std::shared_ptr<energy::Response> force, const Vec& target, double scale)
        : FixedPatternResponse(Eigen::MatrixXd::Ones(force->num_outputs(), force->num_inputs()).sparseView()),
          force_(force), target_(target), scale_(scale) {}

  private:
    Vec eval_impl(const ConstVecRef& theta, SpMat* J) override {
        // Tell Gauss-Newton to reject infeasible trials without changing parameterization.
        if (!theta.allFinite() || theta[1] <= 0 || theta.sum() <= 0)
            throw std::domain_error("Require mu > 0 and lambda_ps + mu > 0");
        Vec value;
        if (J) {
            auto evaluation = force_->evaluate(theta);
            const SpMat& derivative = evaluation->jacobian();
            value = evaluation->value();
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = derivative.coeff(it.row(), c);
        } else {
            value = force_->evaluate(theta)->value();
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

    // Anisotropic stretch, shear and saddle bending excite both material parameters.
    Vec u(map.num_dofs());
    for (int v = 0; v < topology.num_vertices(); ++v) {
        const Eigen::Vector3d X = nodes.row(v).transpose() - center;
        u.segment<3>(3 * v) = Eigen::Vector3d(.08 * X.x() + .04 * X.y(), -.03 * X.y(), .15 * X.x() * X.y() / size);
    }
    const Eigen::Vector2d truth(2, 3), initial(.7, 1.2);
    const fem::ShellFemEnergy elastic(std::make_shared<litepgo::fem::PointwiseShellAdapter<fem::KoiterStVK>>(), map,
                                      rest, thickness, std::vector<Vec>(map.num_elements(), truth));
    // Generate observations through the energy API, independently of the response path.
    const Vec target = elastic.eval_gradient(u);
    const double scale = target.norm();
    if (!(scale > 0) || !std::isfinite(scale))
        throw std::runtime_error("Calibration requires finite nonzero target forces");
    auto force = std::make_shared<fem::ShellFemForceMaterialResponse>(
        std::make_shared<litepgo::fem::PointwiseShellAdapter<fem::KoiterStVK>>(), dof::ConstantDofMap(topology, 2), map,
        rest, u, thickness);
    // Fixed pose and original face areas are cached; only the parameter vector varies.
    if (!force->evaluate(truth)->value().isApprox(target, 1e-11))
        throw std::runtime_error("Fixed-pose response differs from the energy gradient");
    return {std::make_shared<MaterialResidual>(force, target, scale), truth, initial, scale};
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
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-12;
        // Fit the measured response using its analytic parameter Jacobian.
        const auto result = solver::gauss_newton_solve(residual, initial, options);
        const double parameter_error = (result.x - truth).norm() / truth.norm();
        const double force_residual = scale * residual.evaluate(result.x)->value().norm();
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        std::cout << std::setprecision(12) << "truth=" << truth.transpose() << ", initial=" << initial.transpose()
                  << ", fitted=" << result.x.transpose() << ", converged=" << converged
                  << ", iterations=" << result.iterations << ", parameter_error=" << parameter_error
                  << ", force_residual=" << force_residual << ", relative_residual=" << force_residual / scale << '\n';
        return converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shell_fixed_pose_material_fit: " << error.what() << '\n';
        return 1;
    }
}
