#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include "litepgo/fem/shell_fem_force_fp_response.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/gauss_newton.hpp"

#include <Eigen/SVD>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Fit two shared in-plane plastic stretches at a prescribed pose.
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

        // Anisotropic stretch, shear and saddle bending excite both in-plane plastic stretches.
        Vec u(map.num_dofs());
        for (int v = 0; v < topology.num_vertices(); ++v) {
            const Eigen::Vector3d X = nodes.row(v).transpose() - center;
            u.segment<3>(3 * v) = Eigen::Vector3d(.08 * X.x() + .04 * X.y(), -.03 * X.y(), .15 * X.x() * X.y() / size);
        }
        const Eigen::Vector2d truth(1.06, .95), initial = Eigen::Vector2d::Ones();
        const Eigen::Matrix3d true_Fp = Eigen::Vector3d(truth[0], truth[1], 1).asDiagonal();
        const std::vector<Vec> theta(map.num_elements(), Eigen::Vector2d(2, 3));
        // Independent energy gradients: retain x=X+u and ORIGINAL integration areas.
        Vec target = Vec::Zero(map.num_dofs());
        for (int e = 0; e < map.num_elements(); ++e) {
            const Eigen::MatrixXd Y = rest[e].X_local * true_Fp.transpose();
            const fem::TriangleMidEdgeRestData natural(Y, rest[e].opposite);
            const Eigen::MatrixXd natural_u = rest[e].X_local + map.gather(e, u) - Y;
            const fem::ShellMaterialLocalEnergy2D<fem::KoiterStVK> local(natural, natural_u, thickness[e], theta[e]);
            const Vec value = .5 * rest[e].sqrt_det_a_bar * local.eval_gradient(Eigen::Vector2d::Zero());
            const auto indices = map.dof_indices(e);
            for (Eigen::Index i = 0; i < value.size(); ++i)
                target[indices[i]] += value[i];
        }
        const double scale = target.norm();
        if (!(scale > 0) || !std::isfinite(scale))
            throw std::runtime_error("Calibration requires finite nonzero target forces");
        const fem::ShellFemForceFpResponse<fem::KoiterStVK> force(dof::ConstantDofMap(topology, 9), map, rest, u,
                                                                  thickness, theta);

        class PlasticResidual final : public energy::FixedPatternResponse {
          public:
            PlasticResidual(const energy::Response& force, const Vec& target, double scale)
                : FixedPatternResponse(Eigen::MatrixXd::Ones(force.num_outputs(), 2).sparseView()), force_(force),
                  target_(target), scale_(scale) {}

          private:
            Vec eval_values(const ConstVecRef& p, SpMat* J) const override {
                const Eigen::Vector2d stretches = p.array().exp();
                if (!stretches.allFinite() || (stretches.array() <= 0).any())
                    throw std::domain_error("Unrepresentable plastic stretches");
                const Eigen::Matrix3d Fp = Eigen::Vector3d(stretches[0], stretches[1], 1).asDiagonal();
                const Vec packed = Fp.reshaped();
                SpMat full_J;
                if (J)
                    full_J = force_.jacobian_pattern();
                const Vec value = force_.eval(packed, J ? &full_J : nullptr);
                // Column-major vec(Fp): diagonal entries are 0, 4; d(exp(p_j))/dp_j = exp(p_j).
                if (J)
                    for (int j = 0; j < 2; ++j)
                        for (SpMat::InnerIterator it(*J, j); it; ++it)
                            it.valueRef() = full_J.coeff(it.row(), 4 * j) * stretches[j] / scale_;
                return (value - target_) / scale_;
            }
            const energy::Response& force_;
            const Vec& target_;
            double scale_;
        } residual(force, target, scale);
        const auto check_identifiability = [&](const Vec& p) {
            const Eigen::MatrixXd J = residual.eval_jacobian(p);
            const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
            const auto singular = svd.singularValues();
            const double ratio = singular[1] / singular[0];
            if (!(ratio > 1e-6))
                throw std::runtime_error("Prescribed pose cannot locally identify both in-plane plastic stretches");
            return ratio;
        };
        const Vec initial_p = initial.array().log();
        const double initial_ratio = check_identifiability(initial_p);
        const double truth_ratio = check_identifiability(truth.array().log().matrix());
        // Check the composed Jacobian, including the exponential parameterization.
        const Vec check_p = truth.array().log();
        const Eigen::MatrixXd J = residual.eval_jacobian(check_p);
        Eigen::MatrixXd fd(J.rows(), 2);
        for (int j = 0; j < 2; ++j) {
            Vec plus = check_p, minus = check_p;
            plus[j] += 1e-6;
            minus[j] -= 1e-6;
            fd.col(j) = (residual.eval_value(plus) - residual.eval_value(minus)) / 2e-6;
        }
        const double jacobian_error = (J - fd).norm() / J.norm();
        if (!(jacobian_error < 1e-6))
            throw std::runtime_error("Plastic stretch chain-rule check failed");
        solver::GaussNewtonOptions options;
        options.gradient_tolerance = 1e-12;
        const auto result = solver::gauss_newton_solve(residual, initial_p, options);
        const Eigen::Vector2d fitted = result.x.array().exp();
        const double fitted_ratio = check_identifiability(result.x);
        const double parameter_error = (fitted - truth).norm() / truth.norm();
        const double relative_residual = residual.eval_value(result.x).norm();
        const bool converged = result.status == solver::GaussNewtonStatus::Converged;
        std::cout << std::setprecision(12) << "truth=" << truth.transpose() << ", initial=" << initial.transpose()
                  << ", fitted=" << fitted.transpose() << ", converged=" << converged
                  << ", iterations=" << result.iterations << ", parameter_error=" << parameter_error
                  << ", force_residual=" << scale * relative_residual << ", relative_residual=" << relative_residual
                  << ", initial_ratio=" << initial_ratio << ", truth_ratio=" << truth_ratio
                  << ", fitted_ratio=" << fitted_ratio << ", jacobian_error=" << jacobian_error << '\n';
        return converged && parameter_error < 1e-7 && relative_residual < 1e-9 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shell_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
