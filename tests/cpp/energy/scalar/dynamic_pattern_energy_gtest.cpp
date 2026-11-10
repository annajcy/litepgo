#include "../../testing/evaluate_energy.hpp"
#include "litepgo/energy/scalar/dynamic_pattern_energy.hpp"

#include "../../testing/test_energies.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace litepgo::energy::test {

namespace {

// Pairs (0, 1) and (2, 3) active; (1, 2) inactive.
Vec two_pairs_active() {
    return (Vec(4) << 1.0, 1.0, -2.0, 3.0).finished();
}

} // namespace

TEST(DynamicPatternEnergy, EvaluatesCurrentStructureAndOwnsResults) {
    SwitchingEnergy energy(4);
    const Vec x = two_pairs_active();
    const SpMat active = energy.evaluate(x)->hessian();
    const SpMat inactive = energy.evaluate(-Vec::Ones(4))->hessian();
    EXPECT_EQ(active.nonZeros(), 8);
    EXPECT_EQ(inactive.nonZeros(), 4);
    EXPECT_EQ(active.coeff(0, 1), 1);
    EXPECT_EQ(active.coeff(2, 3), 1);
    EXPECT_EQ(Eigen::MatrixXd(active), Eigen::MatrixXd(energy.evaluate(x)->hessian()));
}
} // namespace litepgo::energy::test
