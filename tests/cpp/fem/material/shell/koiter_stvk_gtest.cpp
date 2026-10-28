#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
using namespace litepgo;

ShellMaterialInputs inputs() {
    ShellMaterialInputs p;
    p.a << 2.2, 0.4, 0.4, 1.6;
    p.b << 0.3, -0.1, -0.1, 0.15;
    p.a_bar << 2, 0.3, 0.3, 1.4;
    p.b_bar << 0.1, 0.03, 0.03, -0.2;
    p.thickness = 0.2;
    p.theta = Eigen::Vector2d(2, 3);
    return p;
}

double direct_density(const ShellMaterialInputs& p) {
    const Eigen::Matrix2d inv_a_bar = p.a_bar.inverse();
    const auto psi = [&](const Eigen::Matrix2d& S) {
        const Eigen::Matrix2d M = inv_a_bar * S;
        return 0.5 * p.theta[0] * M.trace() * M.trace() + p.theta[1] * (M * M).trace();
    };
    return p.thickness * psi(0.5 * (p.a - p.a_bar)) + std::pow(p.thickness, 3) / 12 * psi(p.b - p.b_bar);
}

void perturb(ShellMaterialInputs& p, int component, double delta) {
    auto& M = component < 3 ? p.a : p.b;
    switch (component % 3) {
    case 0:
        M(0, 0) += delta;
        break;
    case 1:
        M(0, 1) += delta;
        M(1, 0) += delta;
        break;
    case 2:
        M(1, 1) += delta;
        break;
    }
}

template <class Exception> void expect_invalid(const ShellMaterialInputs& p) {
    EXPECT_THROW(KoiterStVK::eval_W(p), Exception);
    EXPECT_THROW(KoiterStVK::eval_dW_dab(p), Exception);
    EXPECT_THROW(KoiterStVK::eval_d2W_dab2(p), Exception);
}

TEST(KoiterStVK, MatchesInvariantDensityAndSymmetricPackedFiniteDifferences) {
    const auto p = inputs();
    EXPECT_NEAR(KoiterStVK::eval_W(p), direct_density(p), 1e-14);
    const auto g = KoiterStVK::eval_dW_dab(p);
    const auto H = KoiterStVK::eval_d2W_dab2(p);
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-14));
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(H);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0);
    constexpr double step = 1e-6;
    for (int j = 0; j < 6; ++j) {
        auto plus = p, minus = p;
        perturb(plus, j, step);
        perturb(minus, j, -step);
        EXPECT_NEAR(g[j], (direct_density(plus) - direct_density(minus)) / (2 * step), 1e-9);
        const Eigen::Matrix<double, 6, 1> fd =
            (KoiterStVK::eval_dW_dab(plus) - KoiterStVK::eval_dW_dab(minus)) / (2 * step);
        EXPECT_LT((H.col(j) - fd).norm(), 1e-9);
    }
}

TEST(KoiterStVK, HasCorrectMembraneBendingAndMixedEntryFactors) {
    auto p = inputs();
    p.a_bar.setIdentity();
    p.b_bar.setZero();
    p.a = 1.21 * p.a_bar;
    p.b.setZero();
    EXPECT_NEAR(KoiterStVK::eval_W(p), p.thickness / 2 * 5 * 0.21 * 0.21, 1e-14);
    const double W_membrane = KoiterStVK::eval_W(p);
    p.thickness *= 2;
    EXPECT_NEAR(KoiterStVK::eval_W(p), 2 * W_membrane, 1e-14);
    p.a.setIdentity();
    p.b = 0.4 * p.a;
    const double W_bending = KoiterStVK::eval_W(p);
    EXPECT_NEAR(W_bending, std::pow(p.thickness, 3) / 6 * 5 * 0.4 * 0.4, 1e-14);
    p.thickness *= 2;
    EXPECT_NEAR(KoiterStVK::eval_W(p), 8 * W_bending, 1e-14);
    p.b.setZero();
    p.a(0, 1) = p.a(1, 0) = 0.1;
    EXPECT_NEAR(KoiterStVK::eval_W(p), p.thickness * 3 / 2 * 0.1 * 0.1, 1e-14);
    p.a.setIdentity();
    p.b(0, 1) = p.b(1, 0) = 0.1;
    EXPECT_NEAR(KoiterStVK::eval_W(p), std::pow(p.thickness, 3) * 3 / 6 * 0.1 * 0.1, 1e-14);
}

TEST(KoiterStVK, ReferenceStateIsStressFreeAndDensityIsInvariantUnderChartChanges) {
    auto p = inputs();
    const double W = KoiterStVK::eval_W(p);
    Eigen::Matrix2d T;
    T << 1.2, 0.3, -0.2, 0.8;
    for (auto* M : {&p.a, &p.b, &p.a_bar, &p.b_bar})
        *M = (T.transpose() * *M * T).eval();
    EXPECT_NEAR(KoiterStVK::eval_W(p), W, 1e-14);
    p.a = p.a_bar;
    p.b = p.b_bar;
    EXPECT_DOUBLE_EQ(KoiterStVK::eval_W(p), 0);
    EXPECT_TRUE(KoiterStVK::eval_dW_dab(p).isZero());
    auto auxetic = inputs();
    auxetic.theta = Eigen::Vector2d(-0.5, 2);
    EXPECT_GT(KoiterStVK::eval_W(auxetic), 0);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> auxetic_solver(KoiterStVK::eval_d2W_dab2(auxetic));
    EXPECT_GT(auxetic_solver.eigenvalues().minCoeff(), 0);
}

TEST(KoiterStVK, RejectsInvalidParametersFormsAndMetricDomains) {
    const auto p = inputs();
    for (double thickness :
         {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        auto bad = p;
        bad.thickness = thickness;
        expect_invalid<std::invalid_argument>(bad);
    }
    for (int count : {0, 1, 3}) {
        auto bad = p;
        bad.theta = Eigen::VectorXd::Ones(count);
        expect_invalid<std::invalid_argument>(bad);
    }
    for (Eigen::Vector2d theta : {Eigen::Vector2d(1, 0), Eigen::Vector2d(-2, 1), Eigen::Vector2d(-1, 1),
                                  Eigen::Vector2d(1, std::numeric_limits<double>::quiet_NaN())}) {
        auto bad = p;
        bad.theta = theta;
        expect_invalid<std::invalid_argument>(bad);
    }
    for (int component = 0; component < 4; ++component) {
        auto bad = p;
        Eigen::Matrix2d* M[] = {&bad.a, &bad.b, &bad.a_bar, &bad.b_bar};
        (*M[component])(0, 1) += 0.1;
        expect_invalid<std::invalid_argument>(bad);
        (*M[component])(0, 1) = std::numeric_limits<double>::infinity();
        expect_invalid<std::invalid_argument>(bad);
    }
    for (bool reference : {false, true}) {
        auto bad = p;
        (reference ? bad.a_bar : bad.a).setZero();
        expect_invalid<std::domain_error>(bad);
        (reference ? bad.a_bar : bad.a) = -Eigen::Matrix2d::Identity();
        expect_invalid<std::domain_error>(bad);
    }
}

TEST(KoiterStVK, RejectsUnrepresentableStiffnessAndDensity) {
    auto bad = inputs();
    bad.thickness = 1e200;
    expect_invalid<std::domain_error>(bad);
    bad = inputs();
    bad.a_bar = 1e-320 * Eigen::Matrix2d::Identity();
    expect_invalid<std::domain_error>(bad);
    bad = inputs();
    bad.b.setConstant(1e200);
    EXPECT_THROW(KoiterStVK::eval_W(bad), std::domain_error);
}
} // namespace
