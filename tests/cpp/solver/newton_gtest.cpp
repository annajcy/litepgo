#include "litepgo/solver/newton.hpp"

#include "litepgo/energy/quadratic_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"

#include "../testing/test_energies.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using litepgo::DofReduction;
using litepgo::HessianKind;
using litepgo::NewtonOptions;
using litepgo::NewtonResult;
using litepgo::NewtonStatus;
using litepgo::NewtonStep;
using litepgo::SpMat;
using litepgo::Vec;
using litepgo::testing::DoubleWellChain;
using litepgo::testing::HalfSpaceBarrier;
using litepgo::testing::SwitchingEnergy;

SpMat spd_matrix(int n) {
    std::vector<litepgo::Triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 4.0 + i);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    SpMat A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

NewtonOptions options(double tolerance = 1e-10) {
    NewtonOptions result;
    result.gradient_tolerance = tolerance;
    return result;
}

NewtonResult solve(litepgo::Energy& energy, const Vec& x0, std::vector<int> fixed = {},
                   const NewtonOptions& opts = options(), std::vector<NewtonStep>* steps = nullptr) {
    litepgo::AutoIndefiniteSparseSolver solver;
    return litepgo::newton_solve(energy, DofReduction(energy.num_dofs(), std::move(fixed)), x0, opts, solver,
                                 [&](const NewtonStep& step) {
                                     if (steps != nullptr) {
                                         steps->push_back(step);
                                     }
                                 });
}

// Value x^2 / 2 per DOF, but a gradient of the wrong sign: every Newton direction
// raises the energy, so the line search must give up.
class WrongGradientEnergy final : public litepgo::FixedPatternEnergy {
  public:
    explicit WrongGradientEnergy(int n) : FixedPatternEnergy(n, litepgo::testing::chain_pattern(n, {})) {}

  private:
    double evaluate_values(const litepgo::ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind) const override {
        if (gradient != nullptr) {
            *gradient = -x;
        }
        if (hessian != nullptr) {
            std::fill(hessian->valuePtr(), hessian->valuePtr() + hessian->nonZeros(), 1.0);
        }
        return 0.5 * x.squaredNorm();
    }
};

// A finite value and gradient but a NaN Hessian: no diagonal shift can repair it.
class BrokenHessianEnergy final : public litepgo::FixedPatternEnergy {
  public:
    explicit BrokenHessianEnergy(int n) : FixedPatternEnergy(n, litepgo::testing::chain_pattern(n, {})) {}

  private:
    double evaluate_values(const litepgo::ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind) const override {
        if (gradient != nullptr) {
            *gradient = x;
        }
        if (hessian != nullptr) {
            std::fill(hessian->valuePtr(), hessian->valuePtr() + hessian->nonZeros(), std::nan(""));
        }
        return 0.5 * x.squaredNorm();
    }
};

} // namespace

TEST(Newton, SolvesAQuadraticInOneStep) {
    const SpMat A = spd_matrix(6);
    const Vec b = Vec::LinSpaced(6, -1.0, 2.0);
    litepgo::QuadraticEnergy energy(A, b);
    const NewtonResult result = solve(energy, Vec::Zero(6));
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_EQ(result.iterations, 1);
    const Vec expected = Eigen::MatrixXd(A).ldlt().solve(-b);
    EXPECT_LT((result.x - expected).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LE(result.gradient_norm, 1e-10);
}

TEST(Newton, StopsAtOnceWhenAlreadyConverged) {
    litepgo::QuadraticEnergy energy(spd_matrix(4));
    std::vector<NewtonStep> steps;
    const NewtonResult result = solve(energy, Vec::Zero(4), {}, options(), &steps);
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_EQ(result.iterations, 0);
    ASSERT_EQ(steps.size(), 1U);
    EXPECT_EQ(steps.front().iteration, 0);
}

TEST(Newton, KeepsFixedDofsAtTheirInitialValues) {
    const SpMat A = spd_matrix(6);
    const Vec b = Vec::LinSpaced(6, -1.0, 2.0);
    litepgo::QuadraticEnergy energy(A, b);
    Vec x0 = Vec::Zero(6);
    x0[1] = 0.75;
    x0[4] = -1.5;
    const NewtonResult result = solve(energy, x0, {4, 1});
    ASSERT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_EQ(result.x[1], 0.75);
    EXPECT_EQ(result.x[4], -1.5);

    // The free block solves A_ff x_f = -(b_f + A_fc x_c).
    const std::vector<int> free = {0, 2, 3, 5};
    const Eigen::MatrixXd dense = Eigen::MatrixXd(A);
    Eigen::MatrixXd A_ff(4, 4);
    Vec rhs(4);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            A_ff(i, j) = dense(free[i], free[j]);
        }
        rhs[i] = -(b[free[i]] + dense(free[i], 1) * 0.75 + dense(free[i], 4) * -1.5);
    }
    const Vec expected = A_ff.ldlt().solve(rhs);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.x[free[i]], expected[i], 1e-12);
    }
}

TEST(Newton, ConvergesOnANonlinearEnergyFromFarAway) {
    for (const HessianKind kind : {HessianKind::Projected, HessianKind::Exact}) {
        DoubleWellChain energy(6);
        NewtonOptions opts = options(1e-9);
        opts.hessian = kind;
        const NewtonResult result = solve(energy, Vec::LinSpaced(6, 3.0, 4.0), {}, opts);
        EXPECT_EQ(result.status, NewtonStatus::Converged);
        EXPECT_LE(energy.gradient(result.x).cwiseAbs().maxCoeff(), 1e-9);
    }
}

TEST(Newton, ShiftsTheDiagonalForAnIndefiniteExactHessian) {
    // At x = 0.1 the double wells are concave: the exact Hessian is indefinite.
    DoubleWellChain energy(4);
    NewtonOptions opts = options(1e-9);
    opts.hessian = HessianKind::Exact;
    std::vector<NewtonStep> steps;
    const NewtonResult result = solve(energy, Vec::Constant(4, 0.1), {}, opts, &steps);
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    ASSERT_FALSE(steps.empty());
    EXPECT_GT(steps.front().diagonal_shifts, 0);
}

TEST(Newton, StaysFeasibleAgainstABarrier) {
    // A quadratic pulls x_0 towards 3; the barrier keeps it below 1.
    SpMat A(3, 3);
    A.setIdentity();
    const Vec b = (Vec(3) << -3.0, 0.0, 0.0).finished();
    auto energy = litepgo::WeightedEnergySum(
        {{std::make_shared<litepgo::QuadraticEnergy>(A, b), 1.0}, {std::make_shared<HalfSpaceBarrier>(3, 0), 1.0}});
    std::vector<NewtonStep> steps;
    const NewtonResult result = solve(energy, Vec::Zero(3), {}, options(1e-9), &steps);
    ASSERT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_LT(result.x[0], 1.0);
    // Minimizer of x^2/2 - 3x - log(1 - x): x^2 - 4x + 2 = 0, x = 2 - sqrt(2).
    EXPECT_NEAR(result.x[0], 2.0 - std::sqrt(2.0), 1e-9);
    ASSERT_FALSE(steps.empty());
    EXPECT_LT(steps.front().feasible_step, 1.0);
}

TEST(Newton, RejectsAnInfeasibleStart) {
    HalfSpaceBarrier energy(2, 0);
    EXPECT_THROW(solve(energy, Vec::Constant(2, 2.0)), litepgo::InfeasibleState);
}

TEST(Newton, FollowsAChangingPattern) {
    SwitchingEnergy energy(5);
    std::vector<NewtonStep> steps;
    const NewtonResult result =
        solve(energy, (Vec(5) << 1.0, 1.0, -2.0, 3.0, 0.5).finished(), {}, options(1e-10), &steps);
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_LE(result.x.cwiseAbs().maxCoeff(), 1e-9);
    ASSERT_FALSE(steps.empty());
    EXPECT_TRUE(steps.front().pattern_changed);
}

TEST(Newton, ReportsTheIterationLimit) {
    DoubleWellChain energy(3);
    NewtonOptions opts = options();
    opts.max_iterations = 0;
    const NewtonResult result = solve(energy, Vec::Constant(3, 3.0), {}, opts);
    EXPECT_EQ(result.status, NewtonStatus::MaxIterations);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.x, Vec::Constant(3, 3.0));
}

TEST(Newton, ReportsAFailedLineSearch) {
    WrongGradientEnergy energy(3);
    const NewtonResult result = solve(energy, Vec::Ones(3));
    EXPECT_EQ(result.status, NewtonStatus::StepTooSmall);
    EXPECT_EQ(result.x, Vec::Ones(3)); // no step was taken
}

TEST(Newton, ReportsAMissingDescentDirection) {
    BrokenHessianEnergy energy(3);
    std::vector<NewtonStep> steps;
    const NewtonResult result = solve(energy, Vec::Ones(3), {}, options(), &steps);
    EXPECT_EQ(result.status, NewtonStatus::NoDescentDirection);
    ASSERT_EQ(steps.size(), 1U);
    EXPECT_EQ(steps.front().diagonal_shifts, 10);
}

TEST(Newton, WithEveryDofFixedJustEvaluates) {
    litepgo::QuadraticEnergy energy(spd_matrix(3));
    const Vec x0 = Vec::Ones(3);
    const NewtonResult result = solve(energy, x0, {0, 1, 2});
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_EQ(result.x, x0);
    EXPECT_EQ(result.energy, energy.value(x0));
}

TEST(Newton, ChecksItsArguments) {
    litepgo::QuadraticEnergy energy(spd_matrix(3));
    litepgo::AutoSparseSolver solver;
    EXPECT_THROW(litepgo::newton_solve(energy, DofReduction(3, {}), Vec::Zero(3), NewtonOptions(), solver),
                 std::invalid_argument); // no gradient tolerance
    EXPECT_THROW(litepgo::newton_solve(energy, DofReduction(4, {}), Vec::Zero(3), options(), solver),
                 std::invalid_argument);
    NewtonOptions bad_shrink = options();
    bad_shrink.line_search_shrink = 1.0;
    EXPECT_THROW(litepgo::newton_solve(energy, DofReduction(3, {}), Vec::Zero(3), bad_shrink, solver),
                 std::invalid_argument);
}
