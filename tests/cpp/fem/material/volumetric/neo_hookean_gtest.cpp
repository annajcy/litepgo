#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace litepgo::fem::test {

namespace {

TEST(NeoHookean, MixedParameterDerivativeAnalyticDilationAndShear) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Vector2d theta(0, 3);
    const double p = 1.3, stretch = 1.25;
    const auto dilation = NeoHookean::eval_d2W_dF_dtheta({p * stretch * I, p * I, theta});
    Eigen::Matrix<double, 9, 2> expected = Eigen::Matrix<double, 9, 2>::Zero();
    for (int i = 0; i < 3; ++i) {
        expected(4 * i, 0) = p * p * 3 * std::log(stretch) / stretch;
        expected(4 * i, 1) = p * p * (stretch - 1 / stretch);
    }
    EXPECT_TRUE(dilation.isApprox(expected, 1e-13));
    Eigen::Matrix3d F = I;
    F(0, 1) = .4;
    expected.setZero();
    expected(3, 1) = .4;
    expected(1, 1) = .4;
    EXPECT_TRUE(NeoHookean::eval_d2W_dF_dtheta({F, I, theta}).isApprox(expected, 1e-14));
    EXPECT_LT(NeoHookean::eval_d2W_dF_dtheta({p * I, p * I, theta}).norm(), 1e-13);
}

TEST(NeoHookean, MixedDerivativeReconstructsStressWithNontrivialPlasticState) {
    Eigen::Matrix3d F, Fp;
    F << 1.2, .3, -.1, .1, .9, .2, -.05, .1, 1.1;
    Fp << 1.3, .2, .1, 0, .9, -.1, .1, .2, 1.2;
    const Eigen::Matrix3d R = Eigen::AngleAxisd(.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    const auto B = NeoHookean::eval_d2W_dF_dtheta({F, Fp, Eigen::Vector2d(2, 3)});
    for (const Eigen::Vector2d theta : {Eigen::Vector2d(0, 1), Eigen::Vector2d(5, .7), Eigen::Vector2d(2, 3)}) {
        // Exact linear-parameter identity, not a finite-difference approximation.
        const Eigen::Matrix3d P = NeoHookean::eval_dW_dF({F, Fp, theta});
        EXPECT_TRUE((B * theta).isApprox(P.reshaped(), 1e-13));
        EXPECT_EQ((B - NeoHookean::eval_d2W_dF_dtheta({F, Fp, theta})).norm(), 0);
    }
    const auto rotated = NeoHookean::eval_d2W_dF_dtheta({R * F, Fp, Eigen::Vector2d(2, 3)});
    for (int c = 0; c < 2; ++c) {
        const Eigen::Matrix3d expected = R * Eigen::Map<const Eigen::Matrix3d>(B.col(c).data());
        EXPECT_TRUE(rotated.col(c).isApprox(expected.reshaped(), 1e-13));
    }
}

TEST(NeoHookean, MixedDerivativeRejectsInvalidInputsAndOverflow) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Vector2d theta(2, 3);
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({I, I, Eigen::VectorXd::Ones(1)}), std::invalid_argument);
    for (const Eigen::Vector2d bad :
         {Eigen::Vector2d(-1, 3), Eigen::Vector2d(2, 0), Eigen::Vector2d(2, std::numeric_limits<double>::infinity())})
        EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({I, I, bad}), std::invalid_argument);
    Eigen::Matrix3d bad = I;
    bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({I, bad, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({-I, I, theta}), std::domain_error);
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({I, Eigen::Matrix3d::Zero(), theta}), std::domain_error);
    Eigen::Matrix3d Fp = I, F = I;
    Fp(1, 1) = Fp(2, 2) = 1e150;
    F = Fp;
    F(0, 0) = 1e-20;
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({F, Fp, theta}), std::domain_error);
}

TEST(NeoHookean, FiniteInputsWithOverflowingValuesOrDerivativesThrowDomainError) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    VolumetricMaterialInputs input{I, I, Eigen::Vector2d(2, 3)};
    input.F(0, 0) = 1e155;
    ASSERT_TRUE(input.F.allFinite());
    ASSERT_TRUE(std::isfinite(input.F.determinant()));
    EXPECT_THROW(NeoHookean::eval_W(input), std::domain_error);

    input = {I, I, Eigen::Vector2d(0, 1e308)};
    input.F(0, 0) = 2;
    ASSERT_TRUE(std::isfinite(NeoHookean::eval_W(input)));
    EXPECT_THROW(NeoHookean::eval_dW_dF(input), std::domain_error);

    input.F = I;
    ASSERT_DOUBLE_EQ(NeoHookean::eval_W(input), 0);
    ASSERT_TRUE(NeoHookean::eval_dW_dF(input).allFinite());
    EXPECT_THROW(NeoHookean::eval_d2W_dF2(input), std::domain_error);
}

TEST(NeoHookean, StressFreeAndLinearElasticTangent) {
    const Eigen::Vector2d theta(2.0, 3.0);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Fp;
    Fp << 1.3, 0.2, 0.1, 0.0, 0.9, -0.1, 0.1, 0.2, 1.2;
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    for (const Eigen::Matrix3d& F : {Fp, Eigen::Matrix3d(R * Fp)}) {
        EXPECT_NEAR(NeoHookean::eval_W({F, Fp, theta}), 0.0, 1e-13);
        EXPECT_LT(NeoHookean::eval_dW_dF({F, Fp, theta}).norm(), 1e-13);
    }
    const Eigen::Matrix<double, 9, 9> H = NeoHookean::eval_d2W_dF2({I, I, theta});
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
    const double s = 1.25, p = 1.3, log_det_Fe = 3 * std::log(s);
    const double psi = 1.5 * theta[1] * (s * s - 1) - theta[1] * log_det_Fe + 0.5 * theta[0] * log_det_Fe * log_det_Fe;
    EXPECT_NEAR(NeoHookean::eval_W({s * p * I, p * I, theta}), p * p * p * psi, 1e-13);
    const Eigen::Matrix3d expectedP = p * p * (theta[1] * s + (theta[0] * log_det_Fe - theta[1]) / s) * I;
    EXPECT_TRUE(NeoHookean::eval_dW_dF({s * p * I, p * I, theta}).isApprox(expectedP, 1e-13));
    Eigen::Matrix3d F;
    F << 1.2, 0.2, 0.1, -0.1, 0.8, 0.3, 0.2, 0.1, 1.1;
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.6, Eigen::Vector3d(2, -1, 3).normalized()).toRotationMatrix();
    EXPECT_NEAR(NeoHookean::eval_W({R * F, p * I, theta}), NeoHookean::eval_W({F, p * I, theta}), 1e-13);
    EXPECT_TRUE(
        NeoHookean::eval_dW_dF({R * F, p * I, theta}).isApprox(R * NeoHookean::eval_dW_dF({F, p * I, theta}), 1e-13));
}

TEST(NeoHookean, FullGradientAndHessianFiniteDifferences) {
    // Deterministic varied dense states, including compression and non-unit det_Fp.
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
        const Eigen::Matrix3d P = NeoHookean::eval_dW_dF({F, Fp, theta});
        const Eigen::Matrix<double, 9, 9> H = NeoHookean::eval_d2W_dF2({F, Fp, theta});
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
                    (NeoHookean::eval_W({plus, Fp, theta}) - NeoHookean::eval_W({minus, Fp, theta})) / (2 * h);
                const Eigen::Matrix3d dP =
                    (NeoHookean::eval_dW_dF({plus, Fp, theta}) - NeoHookean::eval_dW_dF({minus, Fp, theta})) / (2 * h);
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
            (NeoHookean::eval_W({F + h * U + h * V, Fp, theta}) - NeoHookean::eval_W({F + h * U - h * V, Fp, theta}) -
             NeoHookean::eval_W({F - h * U + h * V, Fp, theta}) + NeoHookean::eval_W({F - h * U - h * V, Fp, theta})) /
            (4 * h * h);
        const double exact = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(U.data()).dot(
            H * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(V.data()));
        EXPECT_NEAR(fd, exact, 2e-6 * (1 + H.norm()));
    }
}

TEST(NeoHookean, RejectsInvalidStatesAndParametersInEveryEntryPoint) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Vector2d theta(2, 3);
    const auto bad_state = [&](const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp) {
        EXPECT_THROW(NeoHookean::eval_W({F, Fp, theta}), std::domain_error);
        EXPECT_THROW(NeoHookean::eval_dW_dF({F, Fp, theta}), std::domain_error);
        EXPECT_THROW(NeoHookean::eval_d2W_dF2({F, Fp, theta}), std::domain_error);
        EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({F, Fp, theta}), std::domain_error);
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
        EXPECT_THROW(NeoHookean::eval_W({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::eval_dW_dF({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::eval_d2W_dF2({I, I, bad}), std::invalid_argument);
        EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({I, I, bad}), std::invalid_argument);
    }
    Eigen::Matrix3d bad = I;
    bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NeoHookean::eval_W({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::eval_dW_dF({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::eval_d2W_dF2({bad, I, theta}), std::invalid_argument);
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dtheta({bad, I, theta}), std::invalid_argument);
}

TEST(NeoHookean, FeasibleStepLimitsBothTotalAndPlasticGradients) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity(), Z = Eigen::Matrix3d::Zero();
    const Eigen::Matrix3d dF = Eigen::Vector3d(-4, -4, 0).asDiagonal();
    const Eigen::Matrix3d dFp = Eigen::Vector3d(-8, 0, 0).asDiagonal();
    const VolumetricMaterialInputs input{I, I, Eigen::Vector2d(2, 3)};
    for (bool plastic : {false, true}) {
        const VolumetricMaterialInputs delta{dF, plastic ? dFp : Z, Eigen::Vector2d::Zero()};
        const double alpha = NeoHookean::eval_feasible_step(input, delta);
        EXPECT_NEAR(alpha, plastic ? 0.1125 : 0.225, 1e-6);
        for (int i = 0; i <= 100; ++i) {
            const double t = alpha * i / 100;
            EXPECT_TRUE(std::isfinite(NeoHookean::eval_W({I + t * delta.F, I + t * delta.Fp, input.theta})));
        }
    }
}

TEST(NeoHookean, FeasibleStepKeepsParametersInTheirDomainIncludingZeroLambda) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity(), Z = Eigen::Matrix3d::Zero();
    VolumetricMaterialInputs input{I, I, Eigen::Vector2d(2, 3)};
    VolumetricMaterialInputs delta{Z, Z, Eigen::Vector2d(-8, 0)};
    EXPECT_NEAR(NeoHookean::eval_feasible_step(input, delta), 0.225, 1e-14);
    delta.theta << 0, -12;
    EXPECT_NEAR(NeoHookean::eval_feasible_step(input, delta), 0.225, 1e-14);
    input.theta[0] = 0;
    delta.theta << -1, 0;
    EXPECT_EQ(NeoHookean::eval_feasible_step(input, delta), 0);
    delta.theta << 1, 1;
    EXPECT_EQ(NeoHookean::eval_feasible_step(input, delta), 1);
    input.theta[0] = 1e308;
    delta.theta << 1e308, 0;
    const double alpha = NeoHookean::eval_feasible_step(input, delta);
    EXPECT_GT(alpha, 0);
    EXPECT_LT(alpha, 1);
    EXPECT_TRUE(std::isfinite(NeoHookean::eval_W({I, I, input.theta + alpha * delta.theta})));
}

TEST(NeoHookean, FeasibleStepRejectsMalformedDirectionsAndInfeasibleStarts) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity(), Z = Eigen::Matrix3d::Zero();
    const VolumetricMaterialInputs input{I, I, Eigen::Vector2d(2, 3)};
    VolumetricMaterialInputs delta{Z, Z, Eigen::Vector2d::Zero()};
    EXPECT_THROW(NeoHookean::eval_feasible_step({-I, I, input.theta}, delta), std::domain_error);
    EXPECT_THROW(NeoHookean::eval_feasible_step({I, Z, input.theta}, delta), std::domain_error);
    EXPECT_THROW(NeoHookean::eval_feasible_step({I, I, Eigen::Vector2d(-1, 3)}, delta), std::invalid_argument);
    delta.theta.resize(1);
    EXPECT_THROW(NeoHookean::eval_feasible_step(input, delta), std::invalid_argument);
    delta.theta = Eigen::Vector2d::Zero();
    delta.Fp(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(NeoHookean::eval_feasible_step(input, delta), std::invalid_argument);
    delta.Fp.setZero();
    delta.theta[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NeoHookean::eval_feasible_step(input, delta), std::invalid_argument);
}

} // namespace

} // namespace litepgo::fem::test

namespace litepgo::fem::test {
TEST(NeoHookean, PlasticMixedDerivativeAnalyticRestAndDilation) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Vector2d theta(2, 3);
    EXPECT_TRUE(NeoHookean::eval_d2W_dF_dFp({I, I, theta}).isApprox(-NeoHookean::eval_d2W_dF2({I, I, theta}), 1e-14));
    const double a = 1.3, b = .8;
    const auto B = NeoHookean::eval_d2W_dF_dFp({a * I, b * I, theta});
    const Eigen::Matrix<double, 9, 1> direction = I.reshaped();
    const double expected = theta[1] * a + b * b / a * (9 * theta[0] * std::log(a / b) - 3 * theta[1] - 3 * theta[0]);
    EXPECT_TRUE((B * direction).isApprox(expected * direction, 1e-13));
}
} // namespace litepgo::fem::test

namespace litepgo::fem::test {
TEST(NeoHookean, PlasticMixedDerivativeMatchesStressPerturbationsAndRejectsInvalidStates) {
    Eigen::Matrix3d F, Fp;
    F << 1.2, .2, -.1, .05, .9, .1, .02, -.04, 1.1;
    Fp << .9, -.1, .03, .04, 1.2, .08, -.02, .05, 1.1;
    const Eigen::Vector2d theta(2, 3);
    const auto B = NeoHookean::eval_d2W_dF_dFp({F, Fp, theta});
    for (int c = 0; c < 9; ++c) {
        auto plus = Fp, minus = Fp;
        plus.data()[c] += 1e-6;
        minus.data()[c] -= 1e-6;
        const Eigen::Matrix3d dP =
            (NeoHookean::eval_dW_dF({F, plus, theta}) - NeoHookean::eval_dW_dF({F, minus, theta})) / 2e-6;
        EXPECT_TRUE(B.col(c).isApprox(dP.reshaped(), 1e-8));
    }
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dFp({F, Eigen::Matrix3d::Zero(), theta}), std::domain_error);
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dFp({-F, Fp, theta}), std::domain_error);
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dFp({F, Fp, Eigen::Vector2d(2, 0)}), std::invalid_argument);
    Fp(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dFp({F, Fp, theta}), std::invalid_argument);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    EXPECT_THROW(NeoHookean::eval_d2W_dF_dFp({I, I, Eigen::Vector2d(0, 1e308)}), std::domain_error);
}
} // namespace litepgo::fem::test
