#include "litepgo/fem/material/multiplicative_kinematics.hpp"
#include "litepgo/energy/energy.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {

MultiplicativeKinematics::MultiplicativeKinematics(const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp) {
    if (!F.allFinite() || !Fp.allFinite()) {
        throw std::invalid_argument("MultiplicativeKinematics requires finite F and Fp");
    }
    det_Fp_ = Fp.determinant();
    if (!std::isfinite(det_Fp_) || det_Fp_ <= 0.0) {
        throw InfeasibleState("MultiplicativeKinematics requires finite det(Fp) > 0");
    }
    inv_Fp_ = Fp.inverse();
    Fe_ = F * inv_Fp_;
    if (!inv_Fp_.allFinite() || !Fe_.allFinite()) {
        throw InfeasibleState("MultiplicativeKinematics: non-finite inverse or elastic gradient");
    }
}

double MultiplicativeKinematics::reference_energy(double psi) const {
    return det_Fp_ * psi;
}

Eigen::Matrix3d MultiplicativeKinematics::dW_dF(const Eigen::Matrix3d& dpsi_dFe) const {
    return det_Fp_ * dpsi_dFe * inv_Fp_.transpose();
}

Eigen::Matrix<double, 9, 9> MultiplicativeKinematics::d2W_dF2(const Eigen::Matrix<double, 9, 9>& d2psi_dFe2) const {
    // vec(delta_F * inv_Fp) = (inv_Fp^T tensor I) vec(delta_F).
    Eigen::Matrix<double, 9, 9> dFe_dF = Eigen::Matrix<double, 9, 9>::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            dFe_dF.block<3, 3>(3 * i, 3 * j).diagonal().setConstant(inv_Fp_(j, i));
        }
    }
    return det_Fp_ * dFe_dF.transpose() * d2psi_dFe2 * dFe_dF;
}

Eigen::Matrix3d MultiplicativeKinematics::dW_dFp(double psi, const Eigen::Matrix3d& dpsi_dFe) const {
    const Eigen::Matrix3d R = psi * Eigen::Matrix3d::Identity() - Fe_.transpose() * dpsi_dFe;
    return det_Fp_ * R * inv_Fp_.transpose();
}

Eigen::Matrix<double, 9, 9> MultiplicativeKinematics::d2W_dFp2(double psi, const Eigen::Matrix3d& dpsi_dFe,
                                                               const Eigen::Matrix<double, 9, 9>& d2psi_dFe2) const {

    const Eigen::Matrix3d& inv_Fp = inv_Fp_;
    const Eigen::Matrix3d R = psi * Eigen::Matrix3d::Identity() - Fe_.transpose() * dpsi_dFe;
    Eigen::Matrix<double, 9, 9> d2W_dFp2;
    for (int column = 0; column < 9; ++column) {
        Eigen::Matrix3d delta_Fp = Eigen::Matrix3d::Zero();
        delta_Fp.data()[column] = 1.0;
        // Differentiate dW_dFp = det_Fp * R * inv_Fp^T in direction delta_Fp.
        const Eigen::Matrix3d delta_Fe = -Fe_ * delta_Fp * inv_Fp;
        const double delta_psi = dpsi_dFe.cwiseProduct(delta_Fe).sum();
        const Eigen::Matrix<double, 9, 1> delta_dpsi_dFe_vec =
            d2psi_dFe2 * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(delta_Fe.data());
        const Eigen::Map<const Eigen::Matrix3d> delta_dpsi_dFe(delta_dpsi_dFe_vec.data());
        const Eigen::Matrix3d delta_R = delta_psi * Eigen::Matrix3d::Identity() - delta_Fe.transpose() * dpsi_dFe -
                                        Fe_.transpose() * delta_dpsi_dFe;
        const Eigen::Matrix3d delta_dW_dFp =
            det_Fp_ * (((inv_Fp * delta_Fp).trace() * R + delta_R) * inv_Fp.transpose() -
                       R * inv_Fp.transpose() * delta_Fp.transpose() * inv_Fp.transpose());
        d2W_dFp2.col(column) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(delta_dW_dFp.data());
    }
    return d2W_dFp2;
}

} // namespace litepgo
