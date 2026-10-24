#include "litepgo/fem/multiplicative_kinematics.hpp"
#include "litepgo/energy/energy.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {

MultiplicativeKinematics::MultiplicativeKinematics(const Mat3& F, const Mat3& Fp) {
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

Mat3 MultiplicativeKinematics::reference_gradient_F(const Mat3& Pe) const {
    return Jp_ * Pe * inverse_Fp_.transpose();
}

Mat9 MultiplicativeKinematics::reference_hessian_F(const Mat9& He) const {
    // vec(dF A) = (A^T tensor I) vec(dF).
    Mat9 T = Mat9::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T.block<3, 3>(3 * i, 3 * j).diagonal().setConstant(inverse_Fp_(j, i));
        }
    }
    return Jp_ * T.transpose() * He * T;
}

Mat3 MultiplicativeKinematics::reference_gradient_Fp(double psi, const Mat3& Pe) const {
    const Mat3 R = psi * Mat3::Identity() - Fe_.transpose() * Pe;
    return Jp_ * R * inverse_Fp_.transpose();
}

Mat9 MultiplicativeKinematics::reference_hessian_Fp(double psi, const Mat3& Pe, const Mat9& He) const {
    using Vec9 = Eigen::Matrix<double, 9, 1>;
    const Mat3& A = inverse_Fp_;
    const Mat3 R = psi * Mat3::Identity() - Fe_.transpose() * Pe;
    Mat9 H;
    for (int column = 0; column < 9; ++column) {
        Mat3 D = Mat3::Zero();
        D.data()[column] = 1.0;
        // Differentiate Gp = Jp R A^T in direction D = delta Fp.
        const Mat3 dFe = -Fe_ * D * A;
        const double dpsi = Pe.cwiseProduct(dFe).sum();
        const Vec9 dPeVector = He * Eigen::Map<const Vec9>(dFe.data());
        const Eigen::Map<const Mat3> dPe(dPeVector.data());
        const Mat3 dR = dpsi * Mat3::Identity() - dFe.transpose() * Pe - Fe_.transpose() * dPe;
        const Mat3 dGp =
            Jp_ * (((A * D).trace() * R + dR) * A.transpose() - R * A.transpose() * D.transpose() * A.transpose());
        H.col(column) = Eigen::Map<const Vec9>(dGp.data());
    }
    return H;
}

} // namespace litepgo
