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

double elastic_value(const Mat3& Fe, const NeoHookean::Parameters& theta, double logJ) {
    const double lambda = theta[0], mu = theta[1];
    return 0.5 * mu * (Fe.squaredNorm() - 3.0) - mu * logJ + 0.5 * lambda * logJ * logJ;
}

Mat3 elastic_gradient(const Mat3& Fe, const NeoHookean::Parameters& theta, double logJ, const Mat3& Q) {
    return theta[1] * Fe + (theta[0] * logJ - theta[1]) * Q;
}

Mat9 elastic_hessian(const NeoHookean::Parameters& theta, double logJ, const Mat3& Q) {
    const double lambda = theta[0], mu = theta[1];
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
    return He;
}

} // namespace

double NeoHookean::value(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_energy(elastic_value(Fe, theta, logJ));
}

Mat3 NeoHookean::gradient_F(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_gradient_F(elastic_gradient(Fe, theta, logJ, inverse_transpose(Fe)));
}

Mat9 NeoHookean::hessian_F(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_hessian_F(elastic_hessian(theta, logJ, inverse_transpose(Fe)));
}

Mat3 NeoHookean::gradient_Fp(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_gradient_Fp(elastic_value(Fe, theta, logJ),
                                   elastic_gradient(Fe, theta, logJ, inverse_transpose(Fe)));
}

Mat9 NeoHookean::hessian_Fp(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    const Mat3 Q = inverse_transpose(Fe);
    return k.reference_hessian_Fp(elastic_value(Fe, theta, logJ), elastic_gradient(Fe, theta, logJ, Q),
                                  elastic_hessian(theta, logJ, Q));
}

NeoHookean::Parameters NeoHookean::gradient_theta(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    const Mat3& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    Parameters gradient;
    gradient << 0.5 * logJ * logJ, 0.5 * (Fe.squaredNorm() - 3.0) - logJ;
    return k.plastic_volume_ratio() * gradient;
}

NeoHookean::ParameterHessian NeoHookean::hessian_theta(const Mat3& F, const Mat3& Fp, const Parameters& theta) const {
    const MultiplicativeKinematics k(F, Fp);
    log_elastic_determinant(k.elastic_gradient(), theta);
    return ParameterHessian::Zero();
}

} // namespace litepgo
