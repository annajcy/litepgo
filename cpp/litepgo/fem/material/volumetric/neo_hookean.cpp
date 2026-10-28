#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/material/volumetric/multiplicative_kinematics.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {
namespace {

double eval_log_det_Fe(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta) {
    if (theta.size() != 2 || !theta.allFinite() || theta[0] < 0.0 || theta[1] <= 0.0) {
        throw std::invalid_argument("NeoHookean requires finite lambda >= 0 and mu > 0");
    }
    const double det_Fe = Fe.determinant();
    if (!std::isfinite(det_Fe) || det_Fe <= 0.0) {
        throw InfeasibleState("NeoHookean requires finite det(Fe) > 0");
    }
    return std::log(det_Fe);
}

Eigen::Matrix3d eval_inv_Fe_T(const Eigen::Matrix3d& Fe) {
    const Eigen::Matrix3d inv_Fe_T = Fe.inverse().transpose();
    if (!inv_Fe_T.allFinite()) {
        throw InfeasibleState("NeoHookean: non-finite elastic inverse");
    }
    return inv_Fe_T;
}

double eval_psi(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta, double log_det_Fe) {
    const double lambda = theta[0], mu = theta[1];
    return 0.5 * mu * (Fe.squaredNorm() - 3.0) - mu * log_det_Fe + 0.5 * lambda * log_det_Fe * log_det_Fe;
}

Eigen::Matrix3d eval_dpsi_dFe(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta, double log_det_Fe,
                              const Eigen::Matrix3d& inv_Fe_T) {
    return theta[1] * Fe + (theta[0] * log_det_Fe - theta[1]) * inv_Fe_T;
}

Eigen::Matrix<double, 9, 9> eval_d2psi_dFe2(const Eigen::VectorXd& theta, double log_det_Fe,
                                            const Eigen::Matrix3d& inv_Fe_T) {
    const double lambda = theta[0], mu = theta[1];
    const double pressure = lambda * log_det_Fe - mu;
    Eigen::Matrix<double, 9, 9> d2psi_dFe2;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    d2psi_dFe2(i + 3 * j, m + 3 * l) = mu * (i == m && j == l) +
                                                       lambda * inv_Fe_T(i, j) * inv_Fe_T(m, l) -
                                                       pressure * inv_Fe_T(i, l) * inv_Fe_T(m, j);
                }
            }
        }
    }
    return d2psi_dFe2;
}

} // namespace

double NeoHookean::eval_W(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    return kinematics.eval_W(eval_psi(Fe, theta, log_det_Fe));
}

Eigen::Matrix3d NeoHookean::eval_dW_dF(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    return kinematics.eval_dW_dF(eval_dpsi_dFe(Fe, theta, log_det_Fe, eval_inv_Fe_T(Fe)));
}

Eigen::Matrix<double, 9, 9> NeoHookean::eval_d2W_dF2(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    return kinematics.eval_d2W_dF2(eval_d2psi_dFe2(theta, log_det_Fe, eval_inv_Fe_T(Fe)));
}

Eigen::Matrix3d NeoHookean::eval_dW_dFp(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    return kinematics.eval_dW_dFp(eval_psi(Fe, theta, log_det_Fe),
                                  eval_dpsi_dFe(Fe, theta, log_det_Fe, eval_inv_Fe_T(Fe)));
}

Eigen::Matrix<double, 9, 9> NeoHookean::eval_d2W_dFp2(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    const Eigen::Matrix3d inv_Fe_T = eval_inv_Fe_T(Fe);
    return kinematics.eval_d2W_dFp2(eval_psi(Fe, theta, log_det_Fe), eval_dpsi_dFe(Fe, theta, log_det_Fe, inv_Fe_T),
                                    eval_d2psi_dFe2(theta, log_det_Fe, inv_Fe_T));
}

Eigen::Vector2d NeoHookean::eval_dW_dtheta(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    Eigen::Vector2d dW_dtheta;
    dW_dtheta << 0.5 * log_det_Fe * log_det_Fe, 0.5 * (Fe.squaredNorm() - 3.0) - log_det_Fe;
    return kinematics.plastic_volume_ratio() * dW_dtheta;
}

Eigen::Matrix2d NeoHookean::eval_d2W_dtheta2(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    eval_log_det_Fe(kinematics.elastic_gradient(), theta);
    return Eigen::Matrix2d::Zero();
}

} // namespace litepgo
