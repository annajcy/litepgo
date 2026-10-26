#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/material/multiplicative_kinematics.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {
namespace {

double log_elastic_determinant(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta) {
    if (theta.size() != 2 || !theta.allFinite() || theta[0] < 0.0 || theta[1] <= 0.0) {
        throw std::invalid_argument("NeoHookean requires finite lambda >= 0 and mu > 0");
    }
    const double det_Fe = Fe.determinant();
    if (!std::isfinite(det_Fe) || det_Fe <= 0.0) {
        throw InfeasibleState("NeoHookean requires finite det(Fe) > 0");
    }
    return std::log(det_Fe);
}

Eigen::Matrix3d inverse_transpose(const Eigen::Matrix3d& Fe) {
    const Eigen::Matrix3d inv_Fe_T = Fe.inverse().transpose();
    if (!inv_Fe_T.allFinite()) {
        throw InfeasibleState("NeoHookean: non-finite elastic inverse");
    }
    return inv_Fe_T;
}

double elastic_value(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta, double log_det_Fe) {
    const double lambda = theta[0], mu = theta[1];
    return 0.5 * mu * (Fe.squaredNorm() - 3.0) - mu * log_det_Fe + 0.5 * lambda * log_det_Fe * log_det_Fe;
}

Eigen::Matrix3d dpsi_dFe(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta, double log_det_Fe,
                         const Eigen::Matrix3d& inv_Fe_T) {
    return theta[1] * Fe + (theta[0] * log_det_Fe - theta[1]) * inv_Fe_T;
}

Eigen::Matrix<double, 9, 9> d2psi_dFe2(const Eigen::VectorXd& theta, double log_det_Fe,
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

double NeoHookean::value(const MaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = log_elastic_determinant(Fe, theta);
    return kinematics.reference_energy(elastic_value(Fe, theta, log_det_Fe));
}

Eigen::Matrix3d NeoHookean::dW_dF(const MaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = log_elastic_determinant(Fe, theta);
    return kinematics.dW_dF(dpsi_dFe(Fe, theta, log_det_Fe, inverse_transpose(Fe)));
}

Eigen::Matrix<double, 9, 9> NeoHookean::d2W_dF2(const MaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = log_elastic_determinant(Fe, theta);
    return kinematics.d2W_dF2(d2psi_dFe2(theta, log_det_Fe, inverse_transpose(Fe)));
}

Eigen::Matrix3d NeoHookean::dW_dFp(const MaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = log_elastic_determinant(Fe, theta);
    return kinematics.dW_dFp(elastic_value(Fe, theta, log_det_Fe),
                             dpsi_dFe(Fe, theta, log_det_Fe, inverse_transpose(Fe)));
}

Eigen::Matrix<double, 9, 9> NeoHookean::d2W_dFp2(const MaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = log_elastic_determinant(Fe, theta);
    const Eigen::Matrix3d inv_Fe_T = inverse_transpose(Fe);
    return kinematics.d2W_dFp2(elastic_value(Fe, theta, log_det_Fe), dpsi_dFe(Fe, theta, log_det_Fe, inv_Fe_T),
                               d2psi_dFe2(theta, log_det_Fe, inv_Fe_T));
}

Eigen::Vector2d NeoHookean::dW_dtheta(const MaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    const Eigen::Matrix3d& Fe = kinematics.elastic_gradient();
    const double log_det_Fe = log_elastic_determinant(Fe, theta);
    Eigen::Vector2d dW_dtheta;
    dW_dtheta << 0.5 * log_det_Fe * log_det_Fe, 0.5 * (Fe.squaredNorm() - 3.0) - log_det_Fe;
    return kinematics.plastic_volume_ratio() * dW_dtheta;
}

Eigen::Matrix2d NeoHookean::d2W_dtheta2(const MaterialInputs& inputs) {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics kinematics(F, Fp);
    log_elastic_determinant(kinematics.elastic_gradient(), theta);
    return Eigen::Matrix2d::Zero();
}

} // namespace litepgo
