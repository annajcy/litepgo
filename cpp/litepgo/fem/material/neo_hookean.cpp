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
    const double Je = Fe.determinant();
    if (!std::isfinite(Je) || Je <= 0.0) {
        throw InfeasibleState("NeoHookean requires finite det(Fe) > 0");
    }
    return std::log(Je);
}

Eigen::Matrix3d inverse_transpose(const Eigen::Matrix3d& Fe) {
    const Eigen::Matrix3d Q = Fe.inverse().transpose();
    if (!Q.allFinite()) {
        throw InfeasibleState("NeoHookean: non-finite elastic inverse");
    }
    return Q;
}

double elastic_value(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta, double logJ) {
    const double lambda = theta[0], mu = theta[1];
    return 0.5 * mu * (Fe.squaredNorm() - 3.0) - mu * logJ + 0.5 * lambda * logJ * logJ;
}

Eigen::Matrix3d elastic_gradient(const Eigen::Matrix3d& Fe, const Eigen::VectorXd& theta, double logJ,
                                 const Eigen::Matrix3d& Q) {
    return theta[1] * Fe + (theta[0] * logJ - theta[1]) * Q;
}

Eigen::Matrix<double, 9, 9> elastic_hessian(const Eigen::VectorXd& theta, double logJ, const Eigen::Matrix3d& Q) {
    const double lambda = theta[0], mu = theta[1];
    const double pressure = lambda * logJ - mu;
    Eigen::Matrix<double, 9, 9> He;
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

double NeoHookean::value(const MaterialInputs& inputs) const {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics k(F, Fp);
    const Eigen::Matrix3d& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_energy(elastic_value(Fe, theta, logJ));
}

Eigen::Matrix3d NeoHookean::gradient_F(const MaterialInputs& inputs) const {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics k(F, Fp);
    const Eigen::Matrix3d& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_gradient_F(elastic_gradient(Fe, theta, logJ, inverse_transpose(Fe)));
}

Eigen::Matrix<double, 9, 9> NeoHookean::hessian_F(const MaterialInputs& inputs) const {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics k(F, Fp);
    const Eigen::Matrix3d& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_hessian_F(elastic_hessian(theta, logJ, inverse_transpose(Fe)));
}

Eigen::Matrix3d NeoHookean::gradient_Fp(const MaterialInputs& inputs) const {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics k(F, Fp);
    const Eigen::Matrix3d& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    return k.reference_gradient_Fp(elastic_value(Fe, theta, logJ),
                                   elastic_gradient(Fe, theta, logJ, inverse_transpose(Fe)));
}

Eigen::Matrix<double, 9, 9> NeoHookean::hessian_Fp(const MaterialInputs& inputs) const {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics k(F, Fp);
    const Eigen::Matrix3d& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    const Eigen::Matrix3d Q = inverse_transpose(Fe);
    return k.reference_hessian_Fp(elastic_value(Fe, theta, logJ), elastic_gradient(Fe, theta, logJ, Q),
                                  elastic_hessian(theta, logJ, Q));
}

Eigen::Vector2d NeoHookean::gradient_theta(const MaterialInputs& inputs) const {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics k(F, Fp);
    const Eigen::Matrix3d& Fe = k.elastic_gradient();
    const double logJ = log_elastic_determinant(Fe, theta);
    Eigen::Vector2d gradient;
    gradient << 0.5 * logJ * logJ, 0.5 * (Fe.squaredNorm() - 3.0) - logJ;
    return k.plastic_volume_ratio() * gradient;
}

Eigen::Matrix2d NeoHookean::hessian_theta(const MaterialInputs& inputs) const {
    const auto& [F, Fp, theta] = inputs;
    const MultiplicativeKinematics k(F, Fp);
    log_elastic_determinant(k.elastic_gradient(), theta);
    return Eigen::Matrix2d::Zero();
}

} // namespace litepgo
