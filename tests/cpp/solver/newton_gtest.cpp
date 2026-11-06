#include "../testing/element_formula_energy.hpp"
#include "litepgo/solver/newton.hpp"

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/energy/scalar/quadratic_energy.hpp"
#include "litepgo/energy/scalar/weighted_sum_energy.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"

#include "../testing/test_energies.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace litepgo::solver::test {

namespace {

using energy::test::DoubleWellChain;
using energy::test::HalfSpaceBarrier;
using energy::test::SwitchingEnergy;
using litepgo::dof::DofReduction;
using litepgo::energy::HessianKind;

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

NewtonResult solve(litepgo::energy::Energy& energy, const Vec& x0, std::vector<int> fixed = {},
                   const NewtonOptions& opts = options(), std::vector<NewtonStep>* steps = nullptr) {
    litepgo::linalg::AutoIndefiniteSparseSolver solver;
    return litepgo::solver::newton_solve(energy, dof::DofReduction(energy.num_dofs(), std::move(fixed)), x0, opts,
                                         solver, [&](const NewtonStep& step) {
                                             if (steps != nullptr) {
                                                 steps->push_back(step);
                                             }
                                         });
}

// Value x^2 / 2 per DOF, but a gradient of the wrong sign: every Newton direction
// raises the energy, so the line search must give up.
class WrongGradientEnergy final : public litepgo::energy::FixedPatternEnergy {
  public:
    explicit WrongGradientEnergy(int n) : energy::FixedPatternEnergy(energy::test::chain_pattern(n, {})) {}

  private:
    double eval_values(const litepgo::ConstVecRef& x, Vec* gradient, SpMat* hessian,
                       energy::HessianKind) const override {
        if (gradient != nullptr) {
            *gradient = -x;
        }
        if (hessian != nullptr) {
            std::fill(hessian->valuePtr(), hessian->valuePtr() + hessian->nonZeros(), 1.0);
        }
        return 0.5 * x.squaredNorm();
    }
};

// A broken energy must be rejected before the linear solve.
class BrokenHessianEnergy final : public litepgo::energy::FixedPatternEnergy {
  public:
    explicit BrokenHessianEnergy(int n) : energy::FixedPatternEnergy(energy::test::chain_pattern(n, {})) {}

  private:
    double eval_values(const litepgo::ConstVecRef& x, Vec* gradient, SpMat* hessian,
                       energy::HessianKind) const override {
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
    litepgo::energy::QuadraticEnergy energy(A, b);
    const NewtonResult result = solve(energy, Vec::Zero(6));
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_EQ(result.iterations, 1);
    const Vec expected = Eigen::MatrixXd(A).ldlt().solve(-b);
    EXPECT_LT((result.x - expected).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LE(result.gradient_norm, 1e-10);
}

TEST(Newton, PropagatesNeoHookeanTrialOverflowWithoutRetrying) {
    const dof::ConstantDofMap map(mesh::TetTopology(4, {{0, 1, 2, 3}}), 9);
    const fem::IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    // A tiny stiffness gives a finite Newton direction whose initial trial overflows ||F||^2.
    const fem::IdentityField3D theta(Eigen::Vector2d(0, 1e-155));
    int trial_evaluations = 0;
    auto elastic = std::make_shared<litepgo::testing::ElementFormulaEnergy>(
        map,
        [Fp, theta, &trial_evaluations](int, const Vec& x, Vec* g, Eigen::MatrixXd* H) {
            if (!g && !H)
                ++trial_evaluations;
            const fem::IdentityField3D F(x);
            const fem::VolumetricMaterialLocalEnergyF3D<fem::NeoHookean> local(F, Fp, theta);
            if (g)
                g->setZero(9);
            if (H)
                H->setZero(9, 9);
            return local.accumulate(Eigen::Vector3d::Zero(), 1, g, H);
        },
        [](int, const Vec& x, const Vec& dx) {
            const fem::VolumetricMaterialInputs input{x.reshaped(3, 3), Eigen::Matrix3d::Identity(),
                                                      Eigen::Vector2d(0, 1e-155)};
            const fem::VolumetricMaterialInputs delta{dx.reshaped(3, 3), Eigen::Matrix3d::Zero(),
                                                      Eigen::Vector2d::Zero()};
            return fem::NeoHookean::eval_feasible_step(input, delta);
        });
    const Vec x0 = Eigen::Matrix3d::Identity().reshaped();
    Vec direction = Vec::Zero(9);
    direction[0] = 5e154;
    EXPECT_DOUBLE_EQ(elastic->max_feasible_step(x0, direction), 1);
    const Vec trial = x0 + direction;
    ASSERT_TRUE(trial.allFinite());
    EXPECT_THROW(elastic->eval_value(trial), std::domain_error);

    Vec load = Vec::Zero(9);
    load[0] = -1;
    auto loading = std::make_shared<energy::QuadraticEnergy>(SpMat(9, 9), load);
    energy::WeightedSumEnergy energy({{elastic, 1}, {loading, 1}});
    ASSERT_TRUE(std::isfinite(energy.eval_value(x0)));
    NewtonOptions opts = options();
    opts.max_iterations = 1;
    opts.hessian = energy::HessianKind::Exact;
    linalg::EigenLDLT solver;
    std::vector<NewtonStep> steps;
    trial_evaluations = 0;
    EXPECT_THROW(newton_solve(energy, dof::DofReduction(9, {1, 2, 3, 4, 5, 6, 7, 8}), x0, opts, solver,
                              [&](const NewtonStep& step) { steps.push_back(step); }),
                 std::domain_error);
    EXPECT_EQ(trial_evaluations, 1);
    EXPECT_TRUE(steps.empty());
}

TEST(Newton, StillBacktracksWhenFiniteTrialEnergyIncreases) {
    DoubleWellChain energy(3);
    const Vec x0 = Vec::Constant(3, 0.6);
    const double initial_energy = energy.eval_value(x0);
    NewtonOptions opts = options();
    opts.max_iterations = 1;
    opts.hessian = energy::HessianKind::Exact;
    std::vector<NewtonStep> steps;
    const auto result = solve(energy, x0, {}, opts, &steps);
    EXPECT_EQ(result.status, NewtonStatus::MaxIterations);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_LT(result.energy, initial_energy);
    ASSERT_FALSE(steps.empty());
    EXPECT_DOUBLE_EQ(steps.front().feasible_step, 1);
    EXPECT_GT(steps.front().step, 0);
    EXPECT_LT(steps.front().step, 1);
    EXPECT_GT(steps.front().line_search_evaluations, 1);
}

TEST(Newton, StopsAtOnceWhenAlreadyConverged) {
    litepgo::energy::QuadraticEnergy energy(spd_matrix(4));
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
    litepgo::energy::QuadraticEnergy energy(A, b);
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
    for (const energy::HessianKind kind : {energy::HessianKind::Projected, energy::HessianKind::Exact}) {
        DoubleWellChain energy(6);
        NewtonOptions opts = options(1e-9);
        opts.hessian = kind;
        const NewtonResult result = solve(energy, Vec::LinSpaced(6, 3.0, 4.0), {}, opts);
        EXPECT_EQ(result.status, NewtonStatus::Converged);
        EXPECT_LE(energy.eval_gradient(result.x).cwiseAbs().maxCoeff(), 1e-9);
    }
}

TEST(Newton, ShiftsTheDiagonalForAnIndefiniteExactHessian) {
    // At x = 0.1 the double wells are concave: the exact Hessian is indefinite.
    DoubleWellChain energy(4);
    NewtonOptions opts = options(1e-9);
    opts.hessian = energy::HessianKind::Exact;
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
    auto energy = litepgo::energy::WeightedSumEnergy({{std::make_shared<litepgo::energy::QuadraticEnergy>(A, b), 1.0},
                                                      {std::make_shared<HalfSpaceBarrier>(3, 0), 1.0}});
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
    EXPECT_THROW(solve(energy, Vec::Constant(2, 2.0)), std::domain_error);
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

TEST(Newton, RejectsAnInvalidHessian) {
    BrokenHessianEnergy energy(3);
    EXPECT_THROW(solve(energy, Vec::Ones(3), {}, options()), std::logic_error);
}

TEST(Newton, WithEveryDofFixedJustEvaluates) {
    litepgo::energy::QuadraticEnergy energy(spd_matrix(3));
    const Vec x0 = Vec::Ones(3);
    const NewtonResult result = solve(energy, x0, {0, 1, 2});
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_EQ(result.x, x0);
    EXPECT_EQ(result.energy, energy.eval_value(x0));
}

TEST(Newton, ChecksItsArguments) {
    litepgo::energy::QuadraticEnergy energy(spd_matrix(3));
    litepgo::linalg::AutoSparseSolver solver;
    EXPECT_THROW(litepgo::solver::newton_solve(energy, dof::DofReduction(3, {}), Vec::Zero(3), NewtonOptions(), solver),
                 std::invalid_argument); // no gradient tolerance
    EXPECT_THROW(litepgo::solver::newton_solve(energy, dof::DofReduction(4, {}), Vec::Zero(3), options(), solver),
                 std::invalid_argument);
    NewtonOptions bad_shrink = options();
    bad_shrink.line_search_shrink = 1.0;
    EXPECT_THROW(litepgo::solver::newton_solve(energy, dof::DofReduction(3, {}), Vec::Zero(3), bad_shrink, solver),
                 std::invalid_argument);
}

} // namespace litepgo::solver::test

TEST(Newton, ReusesSymbolicAnalysisForUnchangedReturnedStructure) {
    using namespace litepgo;
    struct CountingSolver final : linalg::SparseSolver {
        int analyses = 0;
        linalg::EigenLDLT solver;
        bool analyze_pattern(const SpMat& A) override {
            ++analyses;
            return solver.analyze(A);
        }
        bool factor_values(const SpMat& A) override {
            return solver.factorize(A);
        }
        void solve_factored(const ConstVecRef& rhs, VecRef x) const override {
            solver.solve(rhs, x);
        }
    } linear;
    energy::test::DoubleWellChain energy(3);
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-8;
    std::vector<solver::NewtonStep> steps;
    const auto result = solver::newton_solve(energy, dof::DofReduction(3, {}), Vec::Constant(3, 2.), options, linear,
                                             [&](const auto& step) { steps.push_back(step); });
    EXPECT_EQ(result.status, solver::NewtonStatus::Converged);
    ASSERT_GT(steps.size(), 2);
    EXPECT_EQ(linear.analyses, 1);
    EXPECT_TRUE(steps.front().pattern_changed);
    for (size_t i = 1; i < steps.size(); ++i)
        EXPECT_FALSE(steps[i].pattern_changed);
}
