#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_fp_response.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/gauss_newton.hpp"

#include <Eigen/SVD>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Fit three shared plastic stretches at a prescribed pose; no equilibrium or plastic evolution solve.
int main() {
    using namespace litepgo;
    try {
        const auto mesh = std::get<mesh::TetMesh>(mesh::read_veg(LITEPGO_ASSETS_DIR "/box-tet.veg"));
        if (mesh.topology.num_cells() == 0)
            throw std::runtime_error("Example requires a nonempty tetrahedral mesh");
        const dof::VertexDofMap map(mesh.topology, 3);
        using XField = fem::InterpolatedField3D<fem::TetP1Basis>;
        using Field = fem::IdentityField3D;
        std::vector<XField> geometry;
        for (int e = 0; e < map.num_elements(); ++e)
            geometry.emplace_back(map.gather(e, mesh.positions));
        const auto nodes = mesh.positions.reshaped<Eigen::RowMajor>(mesh.topology.num_vertices(), 3);
        const Eigen::Vector3d center = nodes.colwise().mean().transpose();
        Eigen::Matrix3d F;
        F << 1.12, .08, .02, .01, .96, .06, .03, 0, 1.07;
        Vec u(map.num_dofs());
        for (int v = 0; v < mesh.topology.num_vertices(); ++v)
            u.segment<3>(3 * v) = (F - Eigen::Matrix3d::Identity()) * (nodes.row(v).transpose() - center);

        const Eigen::Vector3d truth(1.08, .94, 1.03), initial = Eigen::Vector3d::Ones();
        const Eigen::Matrix3d true_Fp = truth.asDiagonal();
        const std::vector<Field> theta(geometry.size(), Field(Eigen::Vector2d(2, 3)));
        using Elastic = fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature, XField,
                                                 Field, Field>;
        const Elastic elastic(map, geometry, std::vector<Field>(geometry.size(), Field(true_Fp.reshaped())), theta);
        // Target comes from the energy gradient, independently of the force-response path.
        const Vec target = elastic.eval_gradient(u);
        const double scale = target.norm();
        if (!(scale > 0) || !std::isfinite(scale))
            throw std::runtime_error("Calibration requires finite nonzero target forces");
        using Force = fem::VolumetricFemForceFpResponse<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature,
                                                        XField, Field>;
        const Force force(dof::ConstantDofMap(mesh.topology, 9), map, geometry, u, theta);

        class PlasticResidual final : public energy::FixedPatternResponse {
          public:
            PlasticResidual(const energy::Response& force, const Vec& target, double scale)
                : FixedPatternResponse(Eigen::MatrixXd::Ones(force.num_outputs(), 3).sparseView()), force_(force),
                  target_(target), scale_(scale) {}

          private:
            Vec eval_values(const ConstVecRef& p, SpMat* J) const override {
                const Eigen::Vector3d stretches = p.array().exp();
                if (!stretches.allFinite() || (stretches.array() <= 0).any())
                    throw std::domain_error("Unrepresentable plastic stretches");
                const Eigen::Matrix3d Fp = stretches.asDiagonal();
                const Vec packed = Fp.reshaped();
                SpMat full_J;
                if (J)
                    full_J = force_.jacobian_pattern();
                const Vec value = force_.eval(packed, J ? &full_J : nullptr);
                // Column-major vec(Fp): diagonal entries are 0, 4, 8; d(exp(p_j))/dp_j = exp(p_j).
                if (J)
                    for (int j = 0; j < 3; ++j)
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
            const double ratio = singular[2] / singular[0];
            if (!(ratio > 1e-6))
                throw std::runtime_error("Prescribed pose cannot locally identify all three plastic stretches");
            return ratio;
        };
        const Vec initial_p = initial.array().log();
        const double initial_ratio = check_identifiability(initial_p);
        const double truth_ratio = check_identifiability(truth.array().log().matrix());
        // Check the composed Jacobian, including the exponential parameterization.
        const Vec check_p = truth.array().log();
        const Eigen::MatrixXd J = residual.eval_jacobian(check_p);
        Eigen::MatrixXd fd(J.rows(), 3);
        for (int j = 0; j < 3; ++j) {
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
        const Eigen::Vector3d fitted = result.x.array().exp();
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
        std::cerr << "volume_fixed_pose_fp_fit: " << error.what() << '\n';
        return 1;
    }
}
