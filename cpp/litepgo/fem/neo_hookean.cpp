#include "litepgo/fem/neo_hookean.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/multiplicative_kinematics.hpp"

#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {
namespace {

double log_elastic_determinant(const Mat3& Fe, const NeoHookean::Parameters& theta) {
    if (!theta.allFinite() || theta[0] < 0.0 || theta[1] <= 0.0) {
        throw std::invalid_argument("NeoHookean requires finite lambda >= 0 and mu > 0");
    }
    const double Je = Fe.determinant();
    if (!std::isfinite(Je) || Je <= 0.0) {
        throw InfeasibleState("NeoHookean requires finite det(Fe) > 0");
    }
    return std::log(Je);
}

Mat3 inverse_transpose(const Mat3& Fe) {
    const Mat3 Q = Fe.inverse().transpose();
    if (!Q.allFinite()) {
        throw InfeasibleState("NeoHookean: non-finite elastic inverse");
    }
    return Q;
}

} // namespace

double NeoHookean::value(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const double logJ = log_elastic_determinant(k.elastic_gradient(), theta);
    const double lambda = theta[0], mu = theta[1];
    const double psi = 0.5 * mu * (k.elastic_gradient().squaredNorm() - 3.0) - mu * logJ + 0.5 * lambda * logJ * logJ;
    return k.reference_energy(psi);
}

Mat3 NeoHookean::gradient_F(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    const double lambda = theta[0], mu = theta[1];
    const Mat3 Pe = mu * Fe + (lambda * logJ - mu) * inverse_transpose(Fe);
    return k.reference_gradient_F(Pe);
}

Mat9 NeoHookean::hessian_F(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    const double lambda = theta[0], mu = theta[1];
    const Mat3 Q = inverse_transpose(Fe);
    const double pressure = lambda * logJ - mu;
    Mat9 He;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    He(i + 3 * j, m + 3 * l) =
                        mu * (i == m && j == l) + lambda * Q(i, j) * Q(m, l) - pressure * Q(i, l) * Q(m, j);
                }
            }
        }
    }
    return k.reference_hessian_F(He);
}

} // namespace litepgo
