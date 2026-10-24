#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/neo_hookean.hpp"
#include "litepgo/fem/point_energy.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace litepgo {
namespace {

static_assert(PointValue<NeoHookean>);
static_assert(!PointFirstOrderEnergy<NeoHookean>);
static_assert(!PointSecondOrderEnergy<NeoHookean>);
static_assert(PointFirstOrderEnergy<NeoHookeanEnergyF> && PointSecondOrderEnergy<NeoHookeanEnergyF>);
static_assert(PointFirstOrderEnergy<NeoHookeanEnergyFp> && PointSecondOrderEnergy<NeoHookeanEnergyFp>);
static_assert(PointFirstOrderEnergy<NeoHookeanEnergyTheta> && PointSecondOrderEnergy<NeoHookeanEnergyTheta>);

TEST(NeoHookean, StressFreeAndLinearElasticTangent) {
    const NeoHookean m;
    const NeoHookean::Parameters theta(2.0, 3.0);
    const Mat3 I = Mat3::Identity();
    Mat3 Fp;
    Fp << 1.3, 0.2, 0.1, 0.0, 0.9, -0.1, 0.1, 0.2, 1.2;
    const Mat3 R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    for (const Mat3& F : {Fp, Mat3(R * Fp)}) {
        EXPECT_NEAR(m.value({F, Fp, theta}), 0.0, 1e-13);
        EXPECT_LT(NeoHookeanEnergyF{}.gradient({F, Fp, theta}).norm(), 1e-13);
        EXPECT_LT(NeoHookeanEnergyFp{}.gradient({F, Fp, theta}).norm(), 1e-13);
    }
    const Mat9 H = NeoHookeanEnergyF{}.hessian({I, I, theta});
    EXPECT_TRUE(NeoHookeanEnergyFp{}.hessian({I, I, theta}).isApprox(H, 1e-13));
    for (int j = 0; j < 9; ++j) {
        Mat3 D = Mat3::Zero();
        D.data()[j] = 1;
        const Mat3 expected = theta[1] * (D + D.transpose()) + theta[0] * D.trace() * I;
        EXPECT_LT((H.col(j) - Eigen::Map<const Eigen::Matrix<double, 9, 1>>(expected.data())).norm(), 1e-13);
    }
}

TEST(NeoHookean, AnalyticDilationAndObjectivity) {
    const NeoHookean m;
    const NeoHookean::Parameters theta(2, 3);
    const Mat3 I = Mat3::Identity();
    const double s = 1.25, p = 1.3, logJ = 3 * std::log(s);
    const double psi = 1.5 * theta[1] * (s * s - 1) - theta[1] * logJ + 0.5 * theta[0] * logJ * logJ;
    EXPECT_NEAR(m.value({s * p * I, p * I, theta}), p * p * p * psi, 1e-13);
    const Mat3 expectedP = p * p * (theta[1] * s + (theta[0] * logJ - theta[1]) / s) * I;
    EXPECT_TRUE(NeoHookeanEnergyF{}.gradient({s * p * I, p * I, theta}).isApprox(expectedP, 1e-13));
    Mat3 F;
    F << 1.2, 0.2, 0.1, -0.1, 0.8, 0.3, 0.2, 0.1, 1.1;
    const Mat3 R = Eigen::AngleAxisd(0.6, Eigen::Vector3d(2, -1, 3).normalized()).toRotationMatrix();
    EXPECT_NEAR(m.value({R * F, p * I, theta}), m.value({F, p * I, theta}), 1e-13);
    EXPECT_TRUE(NeoHookeanEnergyF{}
                    .gradient({R * F, p * I, theta})
                    .isApprox(R * NeoHookeanEnergyF{}.gradient({F, p * I, theta}), 1e-13));
}

TEST(NeoHookean, FullGradientAndHessianFiniteDifferences) {
    const NeoHookean m;
    // Deterministic varied dense states, including compression and non-unit Jp.
    for (int sample = 0; sample < 12; ++sample) {
        SCOPED_TRACE(sample);
        Mat3 Fp, Fe;
        for (int j = 0; j < 9; ++j) {
            Fp.data()[j] = 0.12 * std::sin(1.7 * (j + 1) * (sample + 1));
            Fe.data()[j] = 0.15 * std::cos(0.8 * (j + 1) * (sample + 2));
        }
        Fp.diagonal().array() += 0.8 + 0.06 * sample;
        Fe.diagonal().array() += 0.7 + 0.05 * sample;
        const Mat3 F = Fe * Fp;
        ASSERT_GT(Fp.determinant(), 0);
        ASSERT_GT(Fe.determinant(), 0);
        const NeoHookean::Point point{F, Fp, NeoHookean::Parameters(2, 3)};
        EXPECT_DOUBLE_EQ(NeoHookeanEnergyF{}.value(point), m.value(point));
        EXPECT_DOUBLE_EQ(NeoHookeanEnergyFp{}.value(point), m.value(point));
        EXPECT_DOUBLE_EQ(NeoHookeanEnergyTheta{}.value(point), m.value(point));
        const NeoHookean::Parameters theta(0.5 * sample, 0.7 + sample);
        const Mat3 P = NeoHookeanEnergyF{}.gradient({F, Fp, theta});
        const Mat9 H = NeoHookeanEnergyF{}.hessian({F, Fp, theta});
        EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
        for (double h : {1e-3, 3e-4, 1e-4}) {
            SCOPED_TRACE(h);
            Mat3 fdP;
            Mat9 fdH;
            for (int j = 0; j < 9; ++j) {
                Mat3 plus = F, minus = F;
                plus.data()[j] += h;
                minus.data()[j] -= h;
                fdP.data()[j] = (m.value({plus, Fp, theta}) - m.value({minus, Fp, theta})) / (2 * h);
                const Mat3 dP = (NeoHookeanEnergyF{}.gradient({plus, Fp, theta}) -
                                 NeoHookeanEnergyF{}.gradient({minus, Fp, theta})) /
                                (2 * h);
                fdH.col(j) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dP.data());
            }
            // O(h^2) central differences, with a roundoff floor.
            EXPECT_LT((P - fdP).norm() / (1 + P.norm()), 20 * h * h + 1e-9);
            EXPECT_LT((H - fdH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
        }
        // Independent Hessian check directly from energy (mixed directions).
        Mat3 U, V;
        U << 1, 2, -1, 0, 1, 3, -2, 0, 1;
        V << 0, 1, 2, -1, 3, 0, 1, -2, 1;
        U.normalize();
        V.normalize();
        const double h = 2e-4;
        const double fd = (m.value({F + h * U + h * V, Fp, theta}) - m.value({F + h * U - h * V, Fp, theta}) -
                           m.value({F - h * U + h * V, Fp, theta}) + m.value({F - h * U - h * V, Fp, theta})) /
                          (4 * h * h);
        const double exact = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(U.data()).dot(
            H * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(V.data()));
        EXPECT_NEAR(fd, exact, 2e-6 * (1 + H.norm()));
    }
}

TEST(NeoHookean, FpGradientAndHessianFiniteDifferences) {
    const NeoHookean m;
    // Deterministic varied dense states, including compression and non-unit Jp.
    for (int sample = 0; sample < 12; ++sample) {
        SCOPED_TRACE(sample);
        Mat3 Fp, Fe;
        for (int j = 0; j < 9; ++j) {
            Fp.data()[j] = 0.12 * std::sin(1.7 * (j + 1) * (sample + 1));
            Fe.data()[j] = 0.15 * std::cos(0.8 * (j + 1) * (sample + 2));
        }
        Fp.diagonal().array() += 0.8 + 0.06 * sample;
        Fe.diagonal().array() += 0.7 + 0.05 * sample;
        const Mat3 F = Fe * Fp;
        ASSERT_GT(Fp.determinant(), 0);
        ASSERT_GT(Fe.determinant(), 0);
        const NeoHookean::Point point{F, Fp, NeoHookean::Parameters(2, 3)};
        EXPECT_DOUBLE_EQ(NeoHookeanEnergyF{}.value(point), m.value(point));
        EXPECT_DOUBLE_EQ(NeoHookeanEnergyFp{}.value(point), m.value(point));
        EXPECT_DOUBLE_EQ(NeoHookeanEnergyTheta{}.value(point), m.value(point));
        const NeoHookean::Parameters theta(0.5 * sample, 0.7 + sample);
        const Mat3 P = NeoHookeanEnergyFp{}.gradient({F, Fp, theta});
        const Mat9 H = NeoHookeanEnergyFp{}.hessian({F, Fp, theta});
        EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
        for (double h : {1e-3, 3e-4, 1e-4}) {
            SCOPED_TRACE(h);
            Mat3 fdP;
            Mat9 fdH;
            for (int j = 0; j < 9; ++j) {
                Mat3 plus = Fp, minus = Fp;
                plus.data()[j] += h;
                minus.data()[j] -= h;
                fdP.data()[j] = (m.value({F, plus, theta}) - m.value({F, minus, theta})) / (2 * h);
                const Mat3 dP = (NeoHookeanEnergyFp{}.gradient({F, plus, theta}) -
                                 NeoHookeanEnergyFp{}.gradient({F, minus, theta})) /
                                (2 * h);
                fdH.col(j) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dP.data());
            }
            // O(h^2) central differences, with a roundoff floor.
            EXPECT_LT((P - fdP).norm() / (1 + P.norm()), 20 * h * h + 1e-9);
            EXPECT_LT((H - fdH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
        }
        // Independent Hessian check directly from energy (mixed directions).
        Mat3 U, V;
        U << 1, 2, -1, 0, 1, 3, -2, 0, 1;
        V << 0, 1, 2, -1, 3, 0, 1, -2, 1;
        U.normalize();
        V.normalize();
        const double h = 2e-4;
        const double fd = (m.value({F, Fp + h * U + h * V, theta}) - m.value({F, Fp + h * U - h * V, theta}) -
                           m.value({F, Fp - h * U + h * V, theta}) + m.value({F, Fp - h * U - h * V, theta})) /
                          (4 * h * h);
        const double exact = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(U.data()).dot(
            H * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(V.data()));
        EXPECT_NEAR(fd, exact, 2e-6 * (1 + H.norm()));
    }
}

TEST(NeoHookean, ParameterDerivativesMatchFiniteDifferences) {
    const NeoHookean m;
    for (int sample = 0; sample < 12; ++sample) {
        SCOPED_TRACE(sample);
        Mat3 Fp, Fe;
        for (int j = 0; j < 9; ++j) {
            Fp.data()[j] = 0.12 * std::sin(1.7 * (j + 1) * (sample + 1));
            Fe.data()[j] = 0.15 * std::cos(0.8 * (j + 1) * (sample + 2));
        }
        Fp.diagonal().array() += 0.8 + 0.06 * sample;
        Fe.diagonal().array() += 0.7 + 0.05 * sample;
        const Mat3 F = Fe * Fp;
        const NeoHookean::Parameters theta(0.5 + sample, 0.7 + sample);
        const auto G = NeoHookeanEnergyTheta{}.gradient({F, Fp, theta});
        const auto H = NeoHookeanEnergyTheta{}.hessian({F, Fp, theta});
        EXPECT_TRUE(H.isZero(0.0));
        for (double h : {1e-2, 1e-3, 1e-4}) {
            SCOPED_TRACE(h);
            for (int j = 0; j < 2; ++j) {
                auto plus = theta, minus = theta;
                plus[j] += h;
                minus[j] -= h;
                const double fd = (m.value({F, Fp, plus}) - m.value({F, Fp, minus})) / (2 * h);
                EXPECT_NEAR(G[j], fd, 1e-9 * (1 + G.norm()));
                const NeoHookean::Parameters dG = (NeoHookeanEnergyTheta{}.gradient({F, Fp, plus}) -
                                                   NeoHookeanEnergyTheta{}.gradient({F, Fp, minus})) /
                                                  (2 * h);
                EXPECT_LT((H.col(j) - dG).norm(), 1e-12);
            }
            // Second differences of energy independently check diagonal and off-diagonal entries.
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    NeoHookean::Parameters u = NeoHookean::Parameters::Zero(), v = u;
                    u[i] = h;
                    v[j] = h;
                    const double fd = (m.value({F, Fp, theta + u + v}) - m.value({F, Fp, theta + u - v}) -
                                       m.value({F, Fp, theta - u + v}) + m.value({F, Fp, theta - u - v})) /
                                      (4 * h * h);
                    EXPECT_NEAR(H(i, j), fd, 1e-6 * (1 + std::abs(m.value({F, Fp, theta}))));
                }
            }
        }
    }
}

TEST(NeoHookean, ParameterDerivativesAtRestAndZeroLambda) {
    const NeoHookean m;
    const Mat3 I = Mat3::Identity();
    const NeoHookean::Parameters theta(0, 3);
    EXPECT_TRUE(NeoHookeanEnergyTheta{}.gradient({I, I, theta}).isZero(0.0));
    EXPECT_TRUE(NeoHookeanEnergyTheta{}.hessian({I, I, theta}).isZero(0.0));
    const Mat3 F = 1.4 * I, Fp = 1.2 * I;
    const auto G = NeoHookeanEnergyTheta{}.gradient({F, Fp, theta});
    const double h = 1e-3;
    auto shifted = theta;
    shifted[0] += h;
    EXPECT_NEAR(G[0], (m.value({F, Fp, shifted}) - m.value({F, Fp, theta})) / h, 1e-10);
    EXPECT_TRUE(NeoHookeanEnergyTheta{}.hessian({F, Fp, theta}).isZero(0.0));
}

TEST(NeoHookean, RejectsInvalidStatesAndParametersInEveryEntryPoint) {
    const NeoHookean m;
    const Mat3 I = Mat3::Identity();
    const NeoHookean::Parameters theta(2, 3);
    const auto bad_state = [&](const Mat3& F, const Mat3& Fp) {
        EXPECT_THROW(m.value({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookeanEnergyF{}.gradient({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookeanEnergyF{}.hessian({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookeanEnergyFp{}.gradient({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookeanEnergyFp{}.hessian({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookeanEnergyTheta{}.gradient({F, Fp, theta}), InfeasibleState);
        EXPECT_THROW(NeoHookeanEnergyTheta{}.hessian({F, Fp, theta}), InfeasibleState);
    };
    for (double x : {0.0, -1.0}) {
        Mat3 bad = I;
        bad(0, 0) = x;
        bad_state(bad, I);
        bad_state(I, bad);
    }
    for (const NeoHookean::Parameters& bad :
         {NeoHookean::Parameters(-1, 3), NeoHookean::Parameters(2, 0), NeoHookean::Parameters(2, -1),
          NeoHookean::Parameters(std::numeric_limits<double>::quiet_NaN(), 3),
          NeoHookean::Parameters(2, std::numeric_limits<double>::infinity())}) {
        EXPECT_THROW(m.value({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookeanEnergyF{}.gradient({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookeanEnergyF{}.hessian({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookeanEnergyFp{}.gradient({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookeanEnergyFp{}.hessian({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookeanEnergyTheta{}.gradient({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookeanEnergyTheta{}.hessian({I, I, bad}), std::invalid_argument);
    }
    Mat3 bad = I;
    bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(m.value({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookeanEnergyF{}.gradient({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookeanEnergyF{}.hessian({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookeanEnergyFp{}.gradient({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookeanEnergyFp{}.hessian({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookeanEnergyTheta{}.gradient({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookeanEnergyTheta{}.hessian({bad, I, theta}), std::invalid_argument);
}

} // namespace
} // namespace litepgo
