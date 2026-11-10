#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"

#include "litepgo/linalg/sparse_pattern.hpp"

#include "../../testing/test_energies.hpp"

#include <gtest/gtest.h>

namespace litepgo::energy::test {

TEST(FixedPatternEnergy, NeverChangesItsPattern) {
    DoubleWellChain energy(4);
    const litepgo::SpMat before = energy.evaluate(Vec::Zero(4))->hessian();

    for (const double scale : {-2.0, 0.0, 0.5, 3.0}) {
        EXPECT_NO_THROW(energy.evaluate(litepgo::Vec::Constant(4, scale))->hessian());
        EXPECT_TRUE(litepgo::linalg::same_pattern(energy.evaluate(Vec::Constant(4, scale))->hessian(), before));
    }
}

} // namespace litepgo::energy::test
