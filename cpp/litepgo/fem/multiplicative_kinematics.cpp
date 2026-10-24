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

} // namespace litepgo
