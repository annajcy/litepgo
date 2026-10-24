#include "litepgo/fem/multiplicative_kinematics.hpp"
#include "litepgo/energy/energy.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {

MultiplicativeKinematics::MultiplicativeKinematics(const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp) {
    if (!F.allFinite() || !Fp.allFinite()) {
        throw std::invalid_argument("MultiplicativeKinematics requires finite F and Fp");
    }
    Jp_ = Fp.determinant();
    if (!std::isfinite(Jp_) || Jp_ <= 0.0) {
        throw InfeasibleState("MultiplicativeKinematics requires finite det(Fp) > 0");
    }
    inverse_Fp_ = Fp.inverse();
    Fe_ = F * inverse_Fp_;
    if (!inverse_Fp_.allFinite() || !Fe_.allFinite()) {
        throw InfeasibleState("MultiplicativeKinematics: non-finite inverse or elastic gradient");
    }
}

double MultiplicativeKinematics::reference_energy(double psi) const {
    return Jp_ * psi;
}

Eigen::Matrix3d MultiplicativeKinematics::reference_gradient_F(const Eigen::Matrix3d& Pe) const {
    return Jp_ * Pe * inverse_Fp_.transpose();
}

Eigen::Matrix<double, 9, 9> MultiplicativeKinematics::reference_hessian_F(const Eigen::Matrix<double, 9, 9>& He) const {
    // vec(dF A) = (A^T tensor I) vec(dF).
    Eigen::Matrix<double, 9, 9> T = Eigen::Matrix<double, 9, 9>::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T.block<3, 3>(3 * i, 3 * j).diagonal().setConstant(inverse_Fp_(j, i));
        }
    }
    return Jp_ * T.transpose() * He * T;
}

Eigen::Matrix3d MultiplicativeKinematics::reference_gradient_Fp(double psi, const Eigen::Matrix3d& Pe) const {
    const Eigen::Matrix3d R = psi * Eigen::Matrix3d::Identity() - Fe_.transpose() * Pe;
    return Jp_ * R * inverse_Fp_.transpose();
}

Eigen::Matrix<double, 9, 9>
MultiplicativeKinematics::reference_hessian_Fp(double psi, const Eigen::Matrix3d& Pe,
                                               const Eigen::Matrix<double, 9, 9>& He) const {

    const Eigen::Matrix3d& A = inverse_Fp_;
    const Eigen::Matrix3d R = psi * Eigen::Matrix3d::Identity() - Fe_.transpose() * Pe;
    Eigen::Matrix<double, 9, 9> H;
    for (int column = 0; column < 9; ++column) {
        Eigen::Matrix3d D = Eigen::Matrix3d::Zero();
        D.data()[column] = 1.0;
        // Differentiate Gp = Jp R A^T in direction D = delta Fp.
        const Eigen::Matrix3d dFe = -Fe_ * D * A;
        const double dpsi = Pe.cwiseProduct(dFe).sum();
        const Eigen::Matrix<double, 9, 1> dPeVector = He * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dFe.data());
        const Eigen::Map<const Eigen::Matrix3d> dPe(dPeVector.data());
        const Eigen::Matrix3d dR = dpsi * Eigen::Matrix3d::Identity() - dFe.transpose() * Pe - Fe_.transpose() * dPe;
        const Eigen::Matrix3d dGp =
            Jp_ * (((A * D).trace() * R + dR) * A.transpose() - R * A.transpose() * D.transpose() * A.transpose());
        H.col(column) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dGp.data());
    }
    return H;
}

} // namespace litepgo
