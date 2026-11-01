#include "litepgo/energy/scalar/weighted_energy_sum.hpp"

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

TEST(WeightedEnergySum, EqualsTheWeightedSumOfItsTerms) {
    const std::vector<WeightedEnergy> terms = {
        {std::make_shared<DoubleWellChain>(5), 2.0},
        {coupling(5, 0, 4), 0.5},
        {std::make_shared<HalfSpaceBarrier>(5, 2), 1.5},
    };
    const WeightedEnergySum sum(terms);
    const Vec x = Vec::LinSpaced(5, -0.8, 0.6);
    expect_weighted_sum(sum, terms, x, HessianKind::Exact);
    expect_weighted_sum(sum, terms, x, HessianKind::Projected);
    expect_derivatives_match(sum, x);
}

TEST(WeightedEnergySum, PatternIsTheUnionOfTheTerms) {
    const WeightedEnergySum sum({{std::make_shared<DoubleWellChain>(5), 1.0}, {coupling(5, 0, 4), 1.0}});
    // Tridiagonal (5 + 2 * 4 entries) plus the (0, 4) and (4, 0) coupling.
    EXPECT_EQ(sum.hessian_pattern().nonZeros(), 15);
    EXPECT_TRUE(stores(sum.hessian_pattern(), 4, 0));
}

TEST(WeightedEnergySum, PatternIsFixedOnlyIfEveryTermsIs) {
    const auto chain = std::make_shared<DoubleWellChain>(4);
    EXPECT_TRUE(WeightedEnergySum({{chain, 1.0}, {coupling(4, 0, 3), 1.0}}).has_fixed_pattern());
    EXPECT_FALSE(WeightedEnergySum({{chain, 1.0}, {std::make_shared<SwitchingEnergy>(4), 1.0}}).has_fixed_pattern());
}

TEST(WeightedEnergySum, FeasibleStepIsTheSmallestTermStep) {
    const WeightedEnergySum sum(
        {{std::make_shared<DoubleWellChain>(3), 1.0}, {std::make_shared<HalfSpaceBarrier>(3, 1), 1.0}});
    const Vec x = Vec::Zero(3);
    const Vec dx = (Vec(3) << 0.0, 2.0, 0.0).finished();
    EXPECT_DOUBLE_EQ(sum.max_feasible_step(x, dx), 0.45); // 0.9 * (1 - 0) / 2
}

TEST(WeightedEnergySum, AllowsNegativeWeights) {
    const std::vector<WeightedEnergy> terms = {{std::make_shared<DoubleWellChain>(4), 1.0}, {coupling(4, 0, 3), -0.5}};
    const WeightedEnergySum sum(terms);
    const Vec x = Vec::LinSpaced(4, -0.9, 0.7);
    expect_weighted_sum(sum, terms, x, HessianKind::Exact);
    expect_derivatives_match(sum, x);
}

TEST(WeightedEnergySum, InfeasibleTermThrowsThroughTheSumWhateverItsWeight) {
    for (const double weight : {2.0, -2.0}) {
        const WeightedEnergySum sum(
            {{std::make_shared<DoubleWellChain>(3), 1.0}, {std::make_shared<HalfSpaceBarrier>(3, 1), weight}});
        const Vec infeasible = (Vec(3) << 0.0, 2.0, 0.0).finished();
        EXPECT_THROW(sum.eval_value(infeasible), std::domain_error);
        Vec gradient;
        EXPECT_THROW(sum.eval(infeasible, &gradient, nullptr), std::domain_error);
    }
}

TEST(WeightedEnergySum, SharedAndNestedTerms) {
    const auto chain = std::make_shared<DoubleWellChain>(4);
    const auto inner = std::make_shared<WeightedEnergySum>(std::vector<WeightedEnergy>{{chain, 1.0}, {chain, 2.0}});
    const WeightedEnergySum outer({{inner, 0.5}, {chain, 1.0}});
    const Vec x = Vec::LinSpaced(4, -1.0, 1.0);
    EXPECT_NEAR(outer.eval_value(x), 2.5 * chain->eval_value(x), 1e-12);
    expect_derivatives_match(outer, x);
}

TEST(WeightedEnergySum, RejectsInvalidTerms) {
    const auto chain = std::make_shared<DoubleWellChain>(3);
    EXPECT_THROW(WeightedEnergySum(std::vector<WeightedEnergy>{}), std::invalid_argument);
    EXPECT_THROW(WeightedEnergySum({{nullptr, 1.0}}), std::invalid_argument);
    EXPECT_THROW(WeightedEnergySum({{chain, 0.0}}), std::invalid_argument);
    EXPECT_THROW(WeightedEnergySum({{chain, std::numeric_limits<double>::infinity()}}), std::invalid_argument);
    EXPECT_THROW(WeightedEnergySum({{chain, std::nan("")}}), std::invalid_argument);
    EXPECT_THROW(WeightedEnergySum({{chain, 1.0}, {std::make_shared<DoubleWellChain>(4), 1.0}}), std::invalid_argument);
}

TEST(WeightedEnergySum, RebuildsWhenATermPatternChanges) {
    const auto switching = std::make_shared<SwitchingEnergy>(4);
    const std::vector<WeightedEnergy> terms = {{std::make_shared<DoubleWellChain>(4), 1.0}, {switching, 3.0}};
    WeightedEnergySum sum(terms);
    const Vec x = (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();

    EXPECT_TRUE(sum.update_pattern(x));
    EXPECT_FALSE(sum.update_pattern(x));
    expect_weighted_sum(sum, terms, x, HessianKind::Exact);
}

TEST(WeightedEnergySum, DetectsATermUpdatedElsewhere) {
    const auto switching = std::make_shared<SwitchingEnergy>(4);
    WeightedEnergySum sum({{switching, 1.0}});
    const Vec x = (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();

    ASSERT_TRUE(switching->update_pattern(x)); // behind the sum's back
    EXPECT_THROW(sum.eval_hessian(x), std::logic_error);
    EXPECT_NO_THROW(sum.eval_value(x));
    // The term already reports no change, but the sum compares patterns and rebuilds.
    EXPECT_TRUE(sum.update_pattern(x));
    EXPECT_NO_THROW(sum.eval_hessian(x));
}

} // namespace litepgo::energy::test
