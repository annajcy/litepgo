#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"

#include "litepgo/linalg/sparse_pattern.hpp"

#include "../../testing/test_energies.hpp"

#include <gtest/gtest.h>

namespace litepgo::energy::test {

TEST(FixedPatternEnergy, NeverChangesItsPattern) {
    DoubleWellChain energy(4);
    const litepgo::SpMat before = energy.eval_hessian(Vec::Zero(4));

    for (const double scale : {-2.0, 0.0, 0.5, 3.0}) {
        EXPECT_NO_THROW(energy.eval_hessian(litepgo::Vec::Constant(4, scale)));
        EXPECT_TRUE(litepgo::linalg::same_pattern(energy.eval_hessian(Vec::Constant(4, scale)), before));
    }
}

} // namespace litepgo::energy::test
