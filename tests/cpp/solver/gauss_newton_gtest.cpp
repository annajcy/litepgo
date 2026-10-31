#include "litepgo/solver/gauss_newton.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace litepgo;
using namespace litepgo::solver;

TEST(GaussNewton, OverdeterminedNonzeroResidual) {
    int values = 0, derivatives = 0;
    ResidualFunction f = [&](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (J) {
            ++derivatives;
            *J = Eigen::Matrix<double, 3, 2>{{1, 0}, {0, 1}, {1, 0}};
        } else
            ++values;
        return (Vec(3) << x[0] - 1, x[1] - 2, x[0] - 3).finished();
    };
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-8;
    const auto result = gauss_newton_solve(f, Vec::Zero(2), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x[0], 2, 1e-8);
    EXPECT_NEAR(result.x[1], 2, 1e-8);
    EXPECT_NEAR(result.objective, 1, 1e-12);
    EXPECT_GT(values, 0);
    EXPECT_EQ(result.evaluations, values + derivatives);
}

TEST(GaussNewton, NonlinearAndInfeasibleTrials) {
    int rejected = 0;
    ResidualFunction f = [&](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (x[0] <= 0) {
            ++rejected;
            throw std::domain_error("positive only");
        }
        if (J)
            *J = Eigen::MatrixXd::Constant(1, 1, 1 / x[0]);
        return Vec::Constant(1, std::log(x[0]));
    };
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    const auto result = gauss_newton_solve(f, Vec::Constant(1, 10), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x[0], 1, 1e-8);
    EXPECT_GT(rejected, 0);
    EXPECT_THROW(gauss_newton_solve(f, Vec::Constant(1, -1), options), std::domain_error);
}

TEST(GaussNewton, LimitsAndValidation) {
    GaussNewtonOptions options;
    ResidualFunction f = [](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Identity(1, 1);
        return x;
    };
    EXPECT_THROW(gauss_newton_solve(f, Vec::Ones(1), options), std::invalid_argument);
    options.gradient_tolerance = 1e-8;
    options.max_iterations = 0;
    EXPECT_EQ(gauss_newton_solve(f, Vec::Ones(1), options).status, GaussNewtonStatus::MaxIterations);
    options.max_iterations = 50;
    ResidualFunction blocked = [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (!J)
            throw std::domain_error("trial blocked");
        *J = Eigen::MatrixXd::Ones(1, 1);
        return Vec::Ones(1);
    };
    EXPECT_EQ(gauss_newton_solve(blocked, Vec::Zero(1), options).status, GaussNewtonStatus::StepTooSmall);
    ResidualFunction bad = [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Zero(2, 2);
        return Vec::Ones(1);
    };
    EXPECT_THROW(gauss_newton_solve(bad, Vec::Zero(1), options), std::logic_error);
    ResidualFunction nan = [](const ConstVecRef&, Eigen::MatrixXd*) -> Vec {
        return Vec::Constant(1, std::numeric_limits<double>::quiet_NaN());
    };
    EXPECT_THROW(gauss_newton_solve(nan, Vec::Zero(1), options), std::logic_error);
}

TEST(GaussNewton, RankDeficientAndCallbackErrors) {
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-8;
    ResidualFunction rank_one = [](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Ones(1, 2);
        return Vec::Constant(1, x.sum() - 2);
    };
    const auto result = gauss_newton_solve(rank_one, Vec::Zero(2), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x.sum(), 2, 1e-8);
    ResidualFunction changed_size = [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Ones(1, 1);
        return Vec::Ones(J ? 1 : 2);
    };
    EXPECT_THROW(gauss_newton_solve(changed_size, Vec::Zero(1), options), std::logic_error);
    ResidualFunction broken_trial = [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (!J)
            throw std::runtime_error("broken callback");
        *J = Eigen::MatrixXd::Ones(1, 1);
        return Vec::Ones(1);
    };
    EXPECT_THROW(gauss_newton_solve(broken_trial, Vec::Zero(1), options), std::runtime_error);
}
