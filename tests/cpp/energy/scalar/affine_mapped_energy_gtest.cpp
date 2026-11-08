#include "litepgo/energy/scalar/affine_mapped_energy.hpp"

#include "../../testing/finite_difference.hpp"
#include "../../testing/test_energies.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace litepgo::energy::test {

namespace {

SpMat from_triplets(int rows, int cols, const std::vector<Triplet>& entries) {
    SpMat T(rows, cols);
    T.setFromTriplets(entries.begin(), entries.end());
    return T;
}

// Columns 0, 2, 3 of the 5 x 5 identity: dofs 1 and 4 held at the offset.
SpMat selection_map() {
    return from_triplets(5, 3, {{0, 0, 1.0}, {2, 1, 1.0}, {3, 2, 1.0}});
}

// Periodic map: dof 4 follows dof 0, so z has 4 entries.
SpMat periodic_map() {
    return from_triplets(5, 4, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}, {3, 3, 1.0}, {4, 0, 1.0}});
}

// A general map with several entries per row.
SpMat general_map() {
    return from_triplets(5, 3,
                         {{0, 0, 1.0},
                          {0, 1, 0.5},
                          {1, 1, -1.0},
                          {2, 0, 0.3},
                          {2, 2, 2.0},
                          {3, 2, 1.0},
                          {4, 0, -0.7},
                          {4, 1, 0.2},
                          {4, 2, 0.4}});
}

void expect_mapped(const AffineMappedEnergy& mapped, const Energy& inner, const SpMat& T, const Vec& offset,
                   const Vec& z, HessianKind kind) {
    const Vec x = T * z + offset;
    const Eigen::MatrixXd dense_T = Eigen::MatrixXd(T);
    EXPECT_NEAR(mapped.eval_value(z), inner.eval_value(x), 1e-12);
    EXPECT_LE((mapped.eval_gradient(z) - dense_T.transpose() * inner.eval_gradient(x)).cwiseAbs().maxCoeff(), 1e-12);
    const Eigen::MatrixXd expected = dense_T.transpose() * Eigen::MatrixXd(inner.eval_hessian(x, kind)) * dense_T;
    EXPECT_LE((Eigen::MatrixXd(mapped.eval_hessian(z, kind)) - expected).cwiseAbs().maxCoeff(), 1e-12);
}

} // namespace

TEST(AffineMappedEnergy, MatchesDenseProductsForEveryKindOfMap) {
    const auto chain = std::make_shared<DoubleWellChain>(5);
    const Vec offset = Vec::LinSpaced(5, 0.1, 0.5);
    for (const SpMat& T : {selection_map(), periodic_map(), general_map()}) {
        const AffineMappedEnergy mapped(chain, T, offset);
        const Vec z = Vec::LinSpaced(T.cols(), -0.7, 0.4);
        expect_mapped(mapped, *chain, T, offset, z, HessianKind::Exact);
        expect_mapped(mapped, *chain, T, offset, z, HessianKind::Projected);
        expect_derivatives_match(mapped, z);
    }
}

TEST(AffineMappedEnergy, DefaultsToAZeroOffset) {
    const auto chain = std::make_shared<DoubleWellChain>(5);
    const AffineMappedEnergy mapped(chain, periodic_map());
    const Vec z = Vec::LinSpaced(4, -0.5, 0.5);
    EXPECT_EQ(mapped.eval_value(z), chain->eval_value(periodic_map() * z));
    EXPECT_EQ(mapped.num_dofs(), 4);
}

TEST(AffineMappedEnergy, ForwardsTheFeasibleStep) {
    const auto barrier = std::make_shared<HalfSpaceBarrier>(5, 4);
    const AffineMappedEnergy mapped(barrier, periodic_map()); // dof 4 follows z_0
    const Vec dz = (Vec(4) << 4.0, 0.0, 0.0, 0.0).finished();
    EXPECT_DOUBLE_EQ(mapped.max_feasible_step(Vec::Zero(4), dz), 0.225); // 0.9 * 1 / 4
}

TEST(AffineMappedEnergy, InfeasibleInnerStateThrowsThrough) {
    const auto barrier = std::make_shared<HalfSpaceBarrier>(5, 4);
    const AffineMappedEnergy mapped(barrier, periodic_map());
    EXPECT_THROW(mapped.eval_value((Vec(4) << 2.0, 0.0, 0.0, 0.0).finished()), std::domain_error);
}

TEST(AffineMappedEnergy, RejectsInvalidInput) {
    const auto chain = std::make_shared<DoubleWellChain>(5);
    EXPECT_THROW(AffineMappedEnergy(nullptr, periodic_map()), std::invalid_argument);
    EXPECT_THROW(AffineMappedEnergy(chain, SpMat(4, 2)), std::invalid_argument);
    EXPECT_THROW(AffineMappedEnergy(chain, periodic_map(), Vec::Zero(3)), std::invalid_argument);
}

TEST(AffineMappedEnergy, RebuildsWhenTheInnerPatternChanges) {
    const auto switching = std::make_shared<SwitchingEnergy>(5);
    const SpMat T = periodic_map();
    const Vec offset = Vec::Zero(5);
    AffineMappedEnergy mapped(switching, T, offset);
    const Vec active = (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();
    const Vec inactive = -Vec::Ones(4);
    const SpMat first = mapped.eval_hessian(active);
    const SpMat second = mapped.eval_hessian(inactive);
    ASSERT_FALSE(linalg::same_pattern(first, second));
    for (const Vec& z : {active, inactive, active})
        expect_mapped(mapped, *switching, T, offset, z, HessianKind::Exact);
    EXPECT_TRUE(linalg::same_pattern(first, mapped.eval_hessian(active)));
}

TEST(AffineMappedEnergy, SharedInnerEvaluationDoesNotInvalidateResults) {
    const auto switching = std::make_shared<SwitchingEnergy>(5);
    const SpMat T = periodic_map();
    AffineMappedEnergy mapped(switching, T);
    AffineMappedEnergy other(switching, T);
    const Vec z = (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();
    const Vec inactive = -Vec::Ones(4);
    const auto saved = mapped.eval(z, true, HessianKind::Exact);
    const Vec gradient = *saved.gradient;
    const SpMat hessian = *saved.hessian;
    const double value = saved.value;

    const SpMat other_hessian = other.eval_hessian(inactive);
    ASSERT_FALSE(linalg::same_pattern(hessian, other_hessian));
    const Eigen::MatrixXd expected = T.transpose() * switching->eval_hessian(T * inactive) * T;
    EXPECT_EQ(Eigen::MatrixXd(other_hessian), expected);
    const auto changed = mapped.eval(inactive, true, HessianKind::Exact);
    EXPECT_NE(changed.value, value);
    EXPECT_EQ(saved.value, value);
    EXPECT_EQ(*saved.gradient, gradient);
    EXPECT_TRUE(linalg::same_pattern(*saved.hessian, hessian));
    EXPECT_EQ(Eigen::MatrixXd(*saved.hessian), Eigen::MatrixXd(hessian));
    const auto again = mapped.eval(z, true, HessianKind::Exact);
    EXPECT_EQ(again.value, value);
    EXPECT_EQ(*again.gradient, gradient);
    EXPECT_TRUE(linalg::same_pattern(*again.hessian, hessian));
    EXPECT_EQ(Eigen::MatrixXd(*again.hessian), Eigen::MatrixXd(hessian));
}

} // namespace litepgo::energy::test
