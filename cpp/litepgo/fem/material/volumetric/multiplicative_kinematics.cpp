#include "litepgo/fem/material/volumetric/multiplicative_kinematics.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem {

MultiplicativeKinematics::MultiplicativeKinematics(const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp) {
    if (!F.allFinite() || !Fp.allFinite()) {
        throw std::invalid_argument("MultiplicativeKinematics requires finite F and Fp");
    }
    det_Fp_ = Fp.determinant();
    if (!std::isfinite(det_Fp_) || det_Fp_ <= 0.0) {
        throw std::domain_error("MultiplicativeKinematics requires finite det(Fp) > 0");
    }
    inv_Fp_ = Fp.inverse();
    Fe_ = F * inv_Fp_;
    if (!inv_Fp_.allFinite() || !Fe_.allFinite()) {
        throw std::domain_error("MultiplicativeKinematics: non-finite inverse or elastic gradient");
    }
}

double MultiplicativeKinematics::eval_W(double psi) const {
    const double W = det_Fp_ * psi;
    if (!std::isfinite(W))
        throw std::domain_error("MultiplicativeKinematics: non-finite density");
    return W;
}

Eigen::Matrix3d MultiplicativeKinematics::eval_dW_dF(const Eigen::Matrix3d& dpsi_dFe) const {
    const Eigen::Matrix3d gradient = det_Fp_ * dpsi_dFe * inv_Fp_.transpose();
    if (!gradient.allFinite())
        throw std::domain_error("MultiplicativeKinematics: non-finite F gradient");
    return gradient;
}

Eigen::Matrix<double, 9, 9>
MultiplicativeKinematics::eval_d2W_dF2(const Eigen::Matrix<double, 9, 9>& d2psi_dFe2) const {
    // vec(delta_F * inv_Fp) = (inv_Fp^T tensor I) vec(delta_F).
    Eigen::Matrix<double, 9, 9> dFe_dF = Eigen::Matrix<double, 9, 9>::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            dFe_dF.block<3, 3>(3 * i, 3 * j).diagonal().setConstant(inv_Fp_(j, i));
        }
    }
    const Eigen::Matrix<double, 9, 9> hessian = det_Fp_ * dFe_dF.transpose() * d2psi_dFe2 * dFe_dF;
    if (!hessian.allFinite())
        throw std::domain_error("MultiplicativeKinematics: non-finite F Hessian");
    return hessian;
}

} // namespace litepgo::fem
