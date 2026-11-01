#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/fem/shell_fem_force_material_response.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/gauss_newton.hpp"

#include <Eigen/SVD>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Fixed-pose force calibration: no equilibrium solve or thickness optimization.
int main() {
    using namespace litepgo;
    try {
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
        const fem::ShellFemEnergy<fem::KoiterStVK> elastic(map, rest, thickness,
                                                           std::vector<Vec>(map.num_elements(), truth));
        // Generate observations through the energy API, independently of the response path.
        const Vec target = elastic.eval_gradient(u);
        const double scale = target.norm();
        if (!(scale > 0) || !std::isfinite(scale))
            throw std::runtime_error("Calibration requires finite nonzero target forces");
        const fem::ShellFemForceMaterialResponse<fem::KoiterStVK> force(dof::ConstantDofMap(topology, 2), map, rest, u,
                                                                        thickness);
        const Eigen::MatrixXd J = force.eval_jacobian(initial);
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
        const auto singular = svd.singularValues();
        const double singular_ratio = singular[1] / singular[0];
        if (!(singular_ratio > 1e-6))
            throw std::runtime_error("Prescribed pose cannot identify both material parameters");

        class MaterialResidual final : public energy::FixedPatternResponse {
          public:
            MaterialResidual(const energy::Response& force, const Vec& target, double scale)
                : FixedPatternResponse(force.jacobian_pattern()), force_(force), target_(target), scale_(scale) {}

          private:
            Vec eval_values(const ConstVecRef& theta, SpMat* J) const override {
                // Tell Gauss-Newton to reject infeasible trials without changing parameterization.
                if (!theta.allFinite() || theta[1] <= 0 || theta.sum() <= 0)
                    throw std::domain_error("Require mu > 0 and lambda_ps + mu > 0");
                const Vec value = force_.eval(theta, J);
                if (J)
                    *J /= scale_;
                return (value - target_) / scale_;
            }
            const energy::Response& force_;
            const Vec& target_;
            double scale_;
        } residual(force, target, scale);
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-12;
        const auto result = solver::gauss_newton_solve(residual, initial, options);
        const double parameter_error = (result.x - truth).norm() / truth.norm();
        const double force_residual = (force.eval_value(result.x) - target).norm();
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        std::cout << std::setprecision(12) << "truth=" << truth.transpose() << ", initial=" << initial.transpose()
                  << ", fitted=" << result.x.transpose() << ", converged=" << converged
                  << ", iterations=" << result.iterations << ", parameter_error=" << parameter_error
                  << ", force_residual=" << force_residual << ", relative_residual=" << force_residual / scale
                  << ", singular_ratio=" << singular_ratio << '\n';
        return converged && parameter_error < 1e-7 && force_residual / scale < 1e-9 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shell_material_fit: " << error.what() << '\n';
        return 1;
    }
}
