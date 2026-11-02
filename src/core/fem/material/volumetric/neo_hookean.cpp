#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/kinematics/positive_det_feasible_step.hpp"
#include "litepgo/fem/material/volumetric/multiplicative_kinematics.hpp"

#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litepgo::fem {
namespace {

double eval_log_det_Fe(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta) {
    if (theta.size() != 2 || !theta.allFinite() || theta[0] < 0.0 || theta[1] <= 0.0) {
        throw std::invalid_argument("NeoHookean requires finite lambda >= 0 and mu > 0");
    }
    const double det_Fe = Fe.determinant();
    if (!std::isfinite(det_Fe) || det_Fe <= 0.0) {
        throw std::domain_error("NeoHookean requires finite det(Fe) > 0");
    }
    return std::log(det_Fe);
}

Eigen::Matrix3d eval_inv_Fe_T(const Eigen::Matrix3d& Fe) {
    const Eigen::Matrix3d inv_Fe_T = Fe.inverse().transpose();
    if (!inv_Fe_T.allFinite()) {
        throw std::domain_error("NeoHookean: non-finite elastic inverse");
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

Eigen::Matrix<double, 9, NeoHookean::num_parameters>
NeoHookean::eval_d2W_dF_dtheta(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    const Eigen::Matrix3d inv_Fe_T = eval_inv_Fe_T(Fe);
    const Eigen::Matrix3d dP_dlambda = kinematics.eval_dW_dF(log_det_Fe * inv_Fe_T);
    const Eigen::Matrix3d dP_dmu = kinematics.eval_dW_dF(Fe - inv_Fe_T);
    Eigen::Matrix<double, 9, num_parameters> result;
    result.col(0) = dP_dlambda.reshaped();
    result.col(1) = dP_dmu.reshaped();
    return result;
}

Eigen::Matrix<double, 9, 9> NeoHookean::eval_d2W_dF_dFp(const VolumetricMaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = eval_log_det_Fe(Fe, theta);
    const Eigen::Matrix3d inv_Fe_T = eval_inv_Fe_T(Fe);
    const Eigen::Matrix3d inv_Fp = Fp.inverse();
    const auto tangent = eval_d2psi_dFe2(theta, log_det_Fe, inv_Fe_T);
    const Eigen::Matrix3d P = kinematics.eval_dW_dF(eval_dpsi_dFe(Fe, theta, log_det_Fe, inv_Fe_T));
    Eigen::Matrix<double, 9, 9> result;
    for (int column = 0; column < 9; ++column) {
        Eigen::Matrix3d delta_Fp = Eigen::Matrix3d::Zero();
        delta_Fp.data()[column] = 1;
        const Eigen::Matrix3d delta_Fe = -Fe * delta_Fp * inv_Fp;
        const Eigen::Matrix<double, 9, 1> delta_stress = tangent * delta_Fe.reshaped();
        // Differentiate P = det(Fp) * psi_Fe * Fp^{-T}, including the volume factor.
        const Eigen::Matrix3d delta_P =
            (inv_Fp * delta_Fp).trace() * P +
            kinematics.plastic_volume_ratio() * delta_stress.reshaped(3, 3) * inv_Fp.transpose() -
            P * delta_Fp.transpose() * inv_Fp.transpose();
        result.col(column) = delta_P.reshaped();
    }
    if (!result.allFinite())
        throw std::domain_error("NeoHookean: non-finite plastic mixed derivative");
    return result;
}

double NeoHookean::eval_feasible_step(const VolumetricMaterialInputs& inputs,
                                      const VolumetricMaterialInputs& delta_inputs) {
    if (!delta_inputs.F.allFinite() || !delta_inputs.Fp.allFinite() || delta_inputs.theta.size() != num_parameters ||
        !delta_inputs.theta.allFinite())
        throw std::invalid_argument("NeoHookean step requires finite, compatible input increments");
    eval_W(inputs);
    double alpha = std::min(eval_positive_det_feasible_step(inputs.F, delta_inputs.F),
                            eval_positive_det_feasible_step(inputs.Fp, delta_inputs.Fp));
    for (int j = 0; j < num_parameters; ++j) {
        if (delta_inputs.theta[j] < 0)
            alpha = std::min(alpha, 0.9 * (inputs.theta[j] / -delta_inputs.theta[j]));
        else if (delta_inputs.theta[j] > 0)
            alpha =
                std::min(alpha, 0.9 * ((std::numeric_limits<double>::max() - inputs.theta[j]) / delta_inputs.theta[j]));
    }
    return alpha;
}

} // namespace litepgo::fem
