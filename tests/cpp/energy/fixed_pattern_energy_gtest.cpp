#include "litepgo/energy/fixed_pattern_energy.h"

#include "litepgo/linalg/sparse_pattern.h"

#include "../testing/test_energies.h"

#include <gtest/gtest.h>

TEST(FixedPatternEnergy, NeverChangesItsPattern) {
    litepgo::testing::DoubleWellChain energy(4);
    const litepgo::SpMat before = energy.hessian_pattern();
    EXPECT_TRUE(energy.has_fixed_pattern());
    for (const double scale : {-2.0, 0.0, 0.5, 3.0}) {
        EXPECT_FALSE(energy.update_pattern(litepgo::Vec::Constant(4, scale)));
        EXPECT_TRUE(litepgo::same_pattern(energy.hessian_pattern(), before));
    }
}
