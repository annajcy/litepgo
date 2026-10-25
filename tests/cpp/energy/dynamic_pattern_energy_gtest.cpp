#include "litepgo/energy/dynamic_pattern_energy.hpp"

#include "../testing/test_energies.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using litepgo::SpMat;
using litepgo::Vec;
using litepgo::testing::SwitchingEnergy;

// Pairs (0, 1) and (2, 3) active; (1, 2) inactive.
Vec two_pairs_active() {
    return (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();
}

} // namespace

TEST(DynamicPatternEnergy, ReportsAChangeOnlyWhenThePatternChanges) {
    SwitchingEnergy energy(4);
    EXPECT_FALSE(energy.has_fixed_pattern());
    EXPECT_EQ(energy.hessian_pattern().nonZeros(), 4);

    EXPECT_TRUE(energy.update_pattern(two_pairs_active()));
    EXPECT_EQ(energy.hessian_pattern().nonZeros(), 8);
    EXPECT_FALSE(energy.update_pattern(two_pairs_active()));
    EXPECT_FALSE(energy.update_pattern(2.0 * two_pairs_active())); // same active set

    EXPECT_TRUE(energy.update_pattern(-Vec::Ones(4)));
    EXPECT_EQ(energy.hessian_pattern().nonZeros(), 4);
}

TEST(DynamicPatternEnergy, HessianNeedsAnUpdatedPattern) {
    SwitchingEnergy energy(4);
    const Vec x = two_pairs_active();
    EXPECT_THROW(energy.hessian(x), std::logic_error);
    EXPECT_NO_THROW(energy.value(x)); // values and gradients never need the pattern
    EXPECT_NO_THROW(energy.gradient(x));

    ASSERT_TRUE(energy.update_pattern(x));
    const SpMat hessian = energy.hessian(x);
    EXPECT_EQ(hessian.coeff(0, 1), 1.0);
    EXPECT_EQ(hessian.coeff(2, 3), 1.0);
}

TEST(DynamicPatternEnergy, OldBuffersAreRejectedAfterAChange) {
    SwitchingEnergy energy(4);
    SpMat buffer = energy.hessian_pattern();
    ASSERT_TRUE(energy.update_pattern(two_pairs_active()));
    EXPECT_THROW(energy.evaluate(two_pairs_active(), nullptr, &buffer), std::invalid_argument);
}
