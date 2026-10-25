#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace litepgo {
namespace {

TEST(NeoHookean, StressFreeAndLinearElasticTangent) {
    const Eigen::Vector2d theta(2.0, 3.0);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Fp;
    Fp << 1.3, 0.2, 0.1, 0.0, 0.9, -0.1, 0.1, 0.2, 1.2;
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    for (const Eigen::Matrix3d& F : {Fp, Eigen::Matrix3d(R * Fp)}) {
        EXPECT_NEAR(NeoHookean::value({F, Fp, theta}), 0.0, 1e-13);
        EXPECT_LT(NeoHookean::gradient_F({F, Fp, theta}).norm(), 1e-13);
        EXPECT_LT(NeoHookean::gradient_Fp({F, Fp, theta}).norm(), 1e-13);
    }
    const Eigen::Matrix<double, 9, 9> H = NeoHookean::hessian_F({I, I, theta});
    EXPECT_TRUE(NeoHookean::hessian_Fp({I, I, theta}).isApprox(H, 1e-13));
    for (int j = 0; j < 9; ++j) {
        Eigen::Matrix3d D = Eigen::Matrix3d::Zero();
        D.data()[j] = 1;
        const Eigen::Matrix3d expected = theta[1] * (D + D.transpose()) + theta[0] * D.trace() * I;
        EXPECT_LT((H.col(j) - Eigen::Map<const Eigen::Matrix<double, 9, 1>>(expected.data())).norm(), 1e-13);
    }
}

TEST(NeoHookean, AnalyticDilationAndObjectivity) {
    const Eigen::Vector2d theta(2, 3);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const double s = 1.25, p = 1.3, logJ = 3 * std::log(s);
    const double psi = 1.5 * theta[1] * (s * s - 1) - theta[1] * logJ + 0.5 * theta[0] * logJ * logJ;
    EXPECT_NEAR(NeoHookean::value({s * p * I, p * I, theta}), p * p * p * psi, 1e-13);
    const Eigen::Matrix3d expectedP = p * p * (theta[1] * s + (theta[0] * logJ - theta[1]) / s) * I;
    EXPECT_TRUE(NeoHookean::gradient_F({s * p * I, p * I, theta}).isApprox(expectedP, 1e-13));
    Eigen::Matrix3d F;
    F << 1.2, 0.2, 0.1, -0.1, 0.8, 0.3, 0.2, 0.1, 1.1;
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.6, Eigen::Vector3d(2, -1, 3).normalized()).toRotationMatrix();
    EXPECT_NEAR(NeoHookean::value({R * F, p * I, theta}), NeoHookean::value({F, p * I, theta}), 1e-13);
    EXPECT_TRUE(
        NeoHookean::gradient_F({R * F, p * I, theta}).isApprox(R * NeoHookean::gradient_F({F, p * I, theta}), 1e-13));
}

TEST(NeoHookean, FullGradientAndHessianFiniteDifferences) {
    // Deterministic varied dense states, including compression and non-unit Jp.
    for (int sample = 0; sample < 12; ++sample) {
        SCOPED_TRACE(sample);
        Eigen::Matrix3d Fp, Fe;
        for (int j = 0; j < 9; ++j) {
            Fp.data()[j] = 0.12 * std::sin(1.7 * (j + 1) * (sample + 1));
            Fe.data()[j] = 0.15 * std::cos(0.8 * (j + 1) * (sample + 2));
        }
        Fp.diagonal().array() += 0.8 + 0.06 * sample;
        Fe.diagonal().array() += 0.7 + 0.05 * sample;
        const Eigen::Matrix3d F = Fe * Fp;
        ASSERT_GT(Fp.determinant(), 0);
        ASSERT_GT(Fe.determinant(), 0);
        const Eigen::Vector2d theta(0.5 * sample, 0.7 + sample);
        const Eigen::Matrix3d P = NeoHookean::gradient_F({F, Fp, theta});
        const Eigen::Matrix<double, 9, 9> H = NeoHookean::hessian_F({F, Fp, theta});
        EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
        for (double h : {1e-3, 3e-4, 1e-4}) {
            SCOPED_TRACE(h);
            Eigen::Matrix3d fdP;
            Eigen::Matrix<double, 9, 9> fdH;
            for (int j = 0; j < 9; ++j) {
                Eigen::Matrix3d plus = F, minus = F;
                plus.data()[j] += h;
                minus.data()[j] -= h;
                fdP.data()[j] =
                    (NeoHookean::value({plus, Fp, theta}) - NeoHookean::value({minus, Fp, theta})) / (2 * h);
                const Eigen::Matrix3d dP =
                    (NeoHookean::gradient_F({plus, Fp, theta}) - NeoHookean::gradient_F({minus, Fp, theta})) / (2 * h);
                fdH.col(j) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dP.data());
            }
            // O(h^2) central differences, with a roundoff floor.
            EXPECT_LT((P - fdP).norm() / (1 + P.norm()), 20 * h * h + 1e-9);
            EXPECT_LT((H - fdH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
        }
        // Independent Hessian check directly from energy (mixed directions).
        Eigen::Matrix3d U, V;
        U << 1, 2, -1, 0, 1, 3, -2, 0, 1;
        V << 0, 1, 2, -1, 3, 0, 1, -2, 1;
        U.normalize();
        V.normalize();
        const double h = 2e-4;
        const double fd =
            (NeoHookean::value({F + h * U + h * V, Fp, theta}) - NeoHookean::value({F + h * U - h * V, Fp, theta}) -
             NeoHookean::value({F - h * U + h * V, Fp, theta}) + NeoHookean::value({F - h * U - h * V, Fp, theta})) /
            (4 * h * h);
        const double exact = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(U.data()).dot(
            H * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(V.data()));
        EXPECT_NEAR(fd, exact, 2e-6 * (1 + H.norm()));
    }
}

TEST(NeoHookean, FpGradientAndHessianFiniteDifferences) {
    // Deterministic varied dense states, including compression and non-unit Jp.
    for (int sample = 0; sample < 12; ++sample) {
        SCOPED_TRACE(sample);
        Eigen::Matrix3d Fp, Fe;
        for (int j = 0; j < 9; ++j) {
            Fp.data()[j] = 0.12 * std::sin(1.7 * (j + 1) * (sample + 1));
            Fe.data()[j] = 0.15 * std::cos(0.8 * (j + 1) * (sample + 2));
        }
        Fp.diagonal().array() += 0.8 + 0.06 * sample;
        Fe.diagonal().array() += 0.7 + 0.05 * sample;
        const Eigen::Matrix3d F = Fe * Fp;
        ASSERT_GT(Fp.determinant(), 0);
        ASSERT_GT(Fe.determinant(), 0);
        const Eigen::Vector2d theta(0.5 * sample, 0.7 + sample);
        const Eigen::Matrix3d P = NeoHookean::gradient_Fp({F, Fp, theta});
        const Eigen::Matrix<double, 9, 9> H = NeoHookean::hessian_Fp({F, Fp, theta});
        EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
        for (double h : {1e-3, 3e-4, 1e-4}) {
            SCOPED_TRACE(h);
            Eigen::Matrix3d fdP;
            Eigen::Matrix<double, 9, 9> fdH;
            for (int j = 0; j < 9; ++j) {
                Eigen::Matrix3d plus = Fp, minus = Fp;
                plus.data()[j] += h;
                minus.data()[j] -= h;
                fdP.data()[j] = (NeoHookean::value({F, plus, theta}) - NeoHookean::value({F, minus, theta})) / (2 * h);
                const Eigen::Matrix3d dP =
                    (NeoHookean::gradient_Fp({F, plus, theta}) - NeoHookean::gradient_Fp({F, minus, theta})) / (2 * h);
                fdH.col(j) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dP.data());
            }
            // O(h^2) central differences, with a roundoff floor.
            EXPECT_LT((P - fdP).norm() / (1 + P.norm()), 20 * h * h + 1e-9);
            EXPECT_LT((H - fdH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
        }
        // Independent Hessian check directly from energy (mixed directions).
        Eigen::Matrix3d U, V;
        U << 1, 2, -1, 0, 1, 3, -2, 0, 1;
        V << 0, 1, 2, -1, 3, 0, 1, -2, 1;
        U.normalize();
        V.normalize();
        const double h = 2e-4;
        const double fd =
            (NeoHookean::value({F, Fp + h * U + h * V, theta}) - NeoHookean::value({F, Fp + h * U - h * V, theta}) -
             NeoHookean::value({F, Fp - h * U + h * V, theta}) + NeoHookean::value({F, Fp - h * U - h * V, theta})) /
            (4 * h * h);
        const double exact = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(U.data()).dot(
            H * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(V.data()));
        EXPECT_NEAR(fd, exact, 2e-6 * (1 + H.norm()));
    }
}

TEST(NeoHookean, ParameterDerivativesMatchFiniteDifferences) {
    for (int sample = 0; sample < 12; ++sample) {
        SCOPED_TRACE(sample);
        Eigen::Matrix3d Fp, Fe;
        for (int j = 0; j < 9; ++j) {
            Fp.data()[j] = 0.12 * std::sin(1.7 * (j + 1) * (sample + 1));
            Fe.data()[j] = 0.15 * std::cos(0.8 * (j + 1) * (sample + 2));
        }
        Fp.diagonal().array() += 0.8 + 0.06 * sample;
        Fe.diagonal().array() += 0.7 + 0.05 * sample;
        const Eigen::Matrix3d F = Fe * Fp;
        const Eigen::Vector2d theta(0.5 + sample, 0.7 + sample);
        const auto G = NeoHookean::gradient_theta({F, Fp, theta});
        const auto H = NeoHookean::hessian_theta({F, Fp, theta});
        EXPECT_TRUE(H.isZero(0.0));
        for (double h : {1e-2, 1e-3, 1e-4}) {
            SCOPED_TRACE(h);
            for (int j = 0; j < 2; ++j) {
                auto plus = theta, minus = theta;
                plus[j] += h;
                minus[j] -= h;
                const double fd = (NeoHookean::value({F, Fp, plus}) - NeoHookean::value({F, Fp, minus})) / (2 * h);
                EXPECT_NEAR(G[j], fd, 1e-9 * (1 + G.norm()));
                const Eigen::Vector2d dG =
                    (NeoHookean::gradient_theta({F, Fp, plus}) - NeoHookean::gradient_theta({F, Fp, minus})) / (2 * h);
                EXPECT_LT((H.col(j) - dG).norm(), 1e-12);
            }
            // Second differences of energy independently check diagonal and off-diagonal entries.
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    Eigen::Vector2d u = Eigen::Vector2d::Zero(), v = u;
                    u[i] = h;
                    v[j] = h;
                    const double fd =
                        (NeoHookean::value({F, Fp, theta + u + v}) - NeoHookean::value({F, Fp, theta + u - v}) -
                         NeoHookean::value({F, Fp, theta - u + v}) + NeoHookean::value({F, Fp, theta - u - v})) /
                        (4 * h * h);
                    EXPECT_NEAR(H(i, j), fd, 1e-6 * (1 + std::abs(NeoHookean::value({F, Fp, theta}))));
                }
            }
        }
    }
}

TEST(NeoHookean, ParameterDerivativesAtRestAndZeroLambda) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Vector2d theta(0, 3);
    EXPECT_TRUE(NeoHookean::gradient_theta({I, I, theta}).isZero(0.0));
    EXPECT_TRUE(NeoHookean::hessian_theta({I, I, theta}).isZero(0.0));
    const Eigen::Matrix3d F = 1.4 * I, Fp = 1.2 * I;
    const auto G = NeoHookean::gradient_theta({F, Fp, theta});
    const double h = 1e-3;
    auto shifted = theta;
    shifted[0] += h;
    EXPECT_NEAR(G[0], (NeoHookean::value({F, Fp, shifted}) - NeoHookean::value({F, Fp, theta})) / h, 1e-10);
    EXPECT_TRUE(NeoHookean::hessian_theta({F, Fp, theta}).isZero(0.0));
}

TEST(NeoHookean, RejectsInvalidStatesAndParametersInEveryEntryPoint) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Vector2d theta(2, 3);
    const auto bad_state = [&](const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp) {
        EXPECT_THROW(NeoHookean::value({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookean::gradient_F({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookean::hessian_F({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookean::gradient_Fp({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookean::hessian_Fp({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookean::gradient_theta({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookean::hessian_theta({F, Fp, theta}), InfeasibleState);
    };
    for (double x : {0.0, -1.0}) {
        Eigen::Matrix3d bad = I;
        bad(0, 0) = x;
        bad_state(bad, I);
        bad_state(I, bad);
    }
    for (const Eigen::Vector2d& bad : {Eigen::Vector2d(-1, 3), Eigen::Vector2d(2, 0), Eigen::Vector2d(2, -1),
                                       Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 3),
                                       Eigen::Vector2d(2, std::numeric_limits<double>::infinity())}) {
        EXPECT_THROW(NeoHookean::value({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::gradient_F({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::hessian_F({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::gradient_Fp({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::hessian_Fp({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::gradient_theta({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::hessian_theta({I, I, bad}), std::invalid_argument);
    }
    Eigen::Matrix3d bad = I;
    bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NeoHookean::value({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::gradient_F({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::hessian_F({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::gradient_Fp({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::hessian_Fp({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::gradient_theta({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::hessian_theta({bad, I, theta}), std::invalid_argument);
}

} // namespace
} // namespace litepgo
