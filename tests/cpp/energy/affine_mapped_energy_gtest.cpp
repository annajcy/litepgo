#include "litepgo/energy/affine_mapped_energy.h"

#include "../testing/finite_difference.h"
#include "../testing/test_energies.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using litepgo::AffineMappedEnergy;
using litepgo::Energy;
using litepgo::HessianKind;
using litepgo::SpMat;
using litepgo::Triplet;
using litepgo::Vec;
using litepgo::testing::DoubleWellChain;
using litepgo::testing::HalfSpaceBarrier;
using litepgo::testing::SwitchingEnergy;

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
    EXPECT_NEAR(mapped.value(z), inner.value(x), 1e-12);
    EXPECT_LE((mapped.gradient(z) - dense_T.transpose() * inner.gradient(x)).cwiseAbs().maxCoeff(), 1e-12);
    const Eigen::MatrixXd expected = dense_T.transpose() * Eigen::MatrixXd(inner.hessian(x, kind)) * dense_T;
    EXPECT_LE((Eigen::MatrixXd(mapped.hessian(z, kind)) - expected).cwiseAbs().maxCoeff(), 1e-12);
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
        litepgo::testing::expect_derivatives_match(mapped, z);
    }
}

TEST(AffineMappedEnergy, DefaultsToAZeroOffset) {
    const auto chain = std::make_shared<DoubleWellChain>(5);
    const AffineMappedEnergy mapped(chain, periodic_map());
    const Vec z = Vec::LinSpaced(4, -0.5, 0.5);
    EXPECT_EQ(mapped.value(z), chain->value(periodic_map() * z));
    EXPECT_EQ(mapped.num_dofs(), 4);
    EXPECT_TRUE(mapped.has_fixed_pattern());
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
    EXPECT_THROW(mapped.value((Vec(4) << 2.0, 0.0, 0.0, 0.0).finished()), litepgo::InfeasibleState);
}

TEST(AffineMappedEnergy, RejectsInvalidInput) {
    const auto chain = std::make_shared<DoubleWellChain>(5);
    EXPECT_THROW(AffineMappedEnergy(nullptr, periodic_map()), std::invalid_argument);
    EXPECT_THROW(AffineMappedEnergy(chain, SpMat(4, 2)), std::invalid_argument);
    EXPECT_THROW(AffineMappedEnergy(chain, periodic_map(), Vec::Zero(3)), std::invalid_argument);
}

TEST(AffineMappedEnergy, RebuildsWhenTheInnerPatternChanges) {
    const auto switching = std::make_shared<SwitchingEnergy>(5);
    const SpMat T = general_map();
    const Vec offset = Vec::Zero(5);
    AffineMappedEnergy mapped(switching, T, offset);
    EXPECT_FALSE(mapped.has_fixed_pattern());

    const Vec z = (Vec(3) << 1.0, -0.5, 0.8).finished();
    ASSERT_FALSE(litepgo::testing::SwitchingEnergy::active_pairs(T * z).empty());
    EXPECT_TRUE(mapped.update_pattern(z));
    EXPECT_FALSE(mapped.update_pattern(z));
    expect_mapped(mapped, *switching, T, offset, z, HessianKind::Exact);
}

TEST(AffineMappedEnergy, DetectsAnInnerEnergyUpdatedElsewhere) {
    const auto switching = std::make_shared<SwitchingEnergy>(5);
    AffineMappedEnergy mapped(switching, periodic_map());
    const Vec z = (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();

    ASSERT_TRUE(switching->update_pattern(periodic_map() * z));
    EXPECT_THROW(mapped.hessian(z), std::logic_error);
    EXPECT_TRUE(mapped.update_pattern(z));
    EXPECT_NO_THROW(mapped.hessian(z));
}
