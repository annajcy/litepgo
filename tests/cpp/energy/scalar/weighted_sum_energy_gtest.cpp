#include "../../testing/evaluate_energy.hpp"
#include "litepgo/energy/scalar/weighted_sum_energy.hpp"

#include "litepgo/energy/scalar/quadratic_energy.hpp"

#include "../../testing/finite_difference.hpp"
#include "../../testing/test_energies.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace litepgo::energy::test {

namespace {

std::shared_ptr<litepgo::energy::QuadraticEnergy> coupling(int n, int i, int j) {
    std::vector<litepgo::Triplet> entries = {{i, i, 1.0}, {j, j, 1.0}, {i, j, -1.0}, {j, i, -1.0}};
    SpMat A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return std::make_shared<litepgo::energy::QuadraticEnergy>(A);
}

void expect_weighted_sum(const Energy& sum, const std::vector<WeightedEnergy>& terms, const Vec& x, HessianKind kind) {
    double value = 0.0;
    Vec gradient = Vec::Zero(x.size());
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(x.size(), x.size());
    for (const WeightedEnergy& term : terms) {
        value += term.weight * term.energy->eval_value(x);
        gradient += term.weight * term.energy->eval_gradient(x);
        hessian += term.weight * Eigen::MatrixXd(term.energy->eval_hessian(x, kind));
    }
    EXPECT_NEAR(sum.eval_value(x), value, 1e-12);
    EXPECT_LE((sum.eval_gradient(x) - gradient).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LE((Eigen::MatrixXd(sum.eval_hessian(x, kind)) - hessian).cwiseAbs().maxCoeff(), 1e-12);
}

} // namespace

TEST(WeightedSumEnergy, EqualsTheWeightedSumOfItsTerms) {
    const std::vector<WeightedEnergy> terms = {
        {std::make_shared<DoubleWellChain>(5), 2.0},
        {coupling(5, 0, 4), 0.5},
        {std::make_shared<HalfSpaceBarrier>(5, 2), 1.5},
    };
    const WeightedSumEnergy sum(terms);
    const Vec x = Vec::LinSpaced(5, -0.8, 0.6);
    expect_weighted_sum(sum, terms, x, HessianKind::Exact);
    expect_weighted_sum(sum, terms, x, HessianKind::Projected);
    expect_derivatives_match(sum, x);
}

TEST(WeightedSumEnergy, PatternIsTheUnionOfTheTerms) {
    const WeightedSumEnergy sum({{std::make_shared<DoubleWellChain>(5), 1.0}, {coupling(5, 0, 4), 1.0}});
    // Tridiagonal (5 + 2 * 4 entries) plus the (0, 4) and (4, 0) coupling.
    EXPECT_EQ(sum.eval_hessian(Vec::Zero(5)).nonZeros(), 15);
    EXPECT_TRUE(stores(sum.eval_hessian(Vec::Zero(5)), 4, 0));
}

TEST(WeightedSumEnergy, FeasibleStepIsTheSmallestTermStep) {
    const WeightedSumEnergy sum(
        {{std::make_shared<DoubleWellChain>(3), 1.0}, {std::make_shared<HalfSpaceBarrier>(3, 1), 1.0}});
    const Vec x = Vec::Zero(3);
    const Vec dx = (Vec(3) << 0.0, 2.0, 0.0).finished();
    EXPECT_DOUBLE_EQ(sum.max_feasible_step(x, dx), 0.45); // 0.9 * (1 - 0) / 2
}

TEST(WeightedSumEnergy, AllowsNegativeWeights) {
    const std::vector<WeightedEnergy> terms = {{std::make_shared<DoubleWellChain>(4), 1.0}, {coupling(4, 0, 3), -0.5}};
    const WeightedSumEnergy sum(terms);
    const Vec x = Vec::LinSpaced(4, -0.9, 0.7);
    expect_weighted_sum(sum, terms, x, HessianKind::Exact);
    expect_derivatives_match(sum, x);
}

TEST(WeightedSumEnergy, InfeasibleTermThrowsThroughTheSumWhateverItsWeight) {
    for (const double weight : {2.0, -2.0}) {
        const WeightedSumEnergy sum(
            {{std::make_shared<DoubleWellChain>(3), 1.0}, {std::make_shared<HalfSpaceBarrier>(3, 1), weight}});
        const Vec infeasible = (Vec(3) << 0.0, 2.0, 0.0).finished();
        EXPECT_THROW(sum.eval_value(infeasible), std::domain_error);
        Vec gradient;
        EXPECT_THROW(litepgo::testing::evaluate_energy(sum, infeasible, &gradient, nullptr), std::domain_error);
    }
}

TEST(WeightedSumEnergy, SharedAndNestedTerms) {
    const auto chain = std::make_shared<DoubleWellChain>(4);
    const auto inner = std::make_shared<WeightedSumEnergy>(std::vector<WeightedEnergy>{{chain, 1.0}, {chain, 2.0}});
    const WeightedSumEnergy outer({{inner, 0.5}, {chain, 1.0}});
    const Vec x = Vec::LinSpaced(4, -1.0, 1.0);
    EXPECT_NEAR(outer.eval_value(x), 2.5 * chain->eval_value(x), 1e-12);
    expect_derivatives_match(outer, x);
}

TEST(WeightedSumEnergy, RejectsInvalidTerms) {
    const auto chain = std::make_shared<DoubleWellChain>(3);
    EXPECT_THROW(WeightedSumEnergy(std::vector<WeightedEnergy>{}), std::invalid_argument);
    EXPECT_THROW(WeightedSumEnergy({{nullptr, 1.0}}), std::invalid_argument);
    EXPECT_THROW(WeightedSumEnergy({{chain, 0.0}}), std::invalid_argument);
    EXPECT_THROW(WeightedSumEnergy({{chain, std::numeric_limits<double>::infinity()}}), std::invalid_argument);
    EXPECT_THROW(WeightedSumEnergy({{chain, std::nan("")}}), std::invalid_argument);
    EXPECT_THROW(WeightedSumEnergy({{chain, 1.0}, {std::make_shared<DoubleWellChain>(4), 1.0}}), std::invalid_argument);
}

TEST(WeightedSumEnergy, RebuildsWhenATermPatternChanges) {
    const auto switching = std::make_shared<SwitchingEnergy>(4);
    const std::vector<WeightedEnergy> terms = {{coupling(4, 0, 3), 1.0}, {switching, 3.0}};
    WeightedSumEnergy sum(terms);
    const Vec active = (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();
    const Vec inactive = -Vec::Ones(4);
    const SpMat first = sum.eval_hessian(active);
    const SpMat second = sum.eval_hessian(inactive);
    ASSERT_FALSE(linalg::same_pattern(first, second));
    for (const Vec& x : {active, inactive, active})
        expect_weighted_sum(sum, terms, x, HessianKind::Exact);
    EXPECT_TRUE(linalg::same_pattern(first, sum.eval_hessian(active)));
}

TEST(WeightedSumEnergy, SharedTermEvaluationDoesNotInvalidateResults) {
    const auto switching = std::make_shared<SwitchingEnergy>(4);
    WeightedSumEnergy sum({{switching, 1.0}});
    WeightedSumEnergy other({{switching, 2.0}});
    const Vec x = (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();
    const Vec inactive = -Vec::Ones(4);
    const auto saved = sum.eval(x, true, HessianKind::Exact);
    const Vec gradient = *saved.gradient;
    const SpMat hessian = *saved.hessian;
    const double value = saved.value;

    EXPECT_EQ(other.eval_value(inactive), 2 * switching->eval_value(inactive));
    const SpMat other_hessian = other.eval_hessian(inactive);
    ASSERT_FALSE(linalg::same_pattern(hessian, other_hessian));
    EXPECT_EQ(Eigen::MatrixXd(other_hessian), 2 * Eigen::MatrixXd(switching->eval_hessian(inactive)));
    const auto changed = sum.eval(inactive, true, HessianKind::Exact);
    EXPECT_NE(changed.value, value);
    EXPECT_EQ(saved.value, value);
    EXPECT_EQ(*saved.gradient, gradient);
    EXPECT_TRUE(linalg::same_pattern(*saved.hessian, hessian));
    EXPECT_EQ(Eigen::MatrixXd(*saved.hessian), Eigen::MatrixXd(hessian));
    const auto again = sum.eval(x, true, HessianKind::Exact);
    EXPECT_EQ(again.value, value);
    EXPECT_EQ(*again.gradient, gradient);
    EXPECT_TRUE(linalg::same_pattern(*again.hessian, hessian));
    EXPECT_EQ(Eigen::MatrixXd(*again.hessian), Eigen::MatrixXd(hessian));
}

} // namespace litepgo::energy::test
