#include "litepgo/core/memory.hpp"

#include <gtest/gtest.h>

#include <array>

namespace litepgo::test {

TEST(Memory, Overlaps) {
    std::array<double, 10> data{};
    const double* p = data.data();
    EXPECT_TRUE(litepgo::overlaps(p, 5, p, 5));
    EXPECT_TRUE(litepgo::overlaps(p, 5, p + 4, 3));
    EXPECT_TRUE(litepgo::overlaps(p + 4, 3, p, 5));
    EXPECT_FALSE(litepgo::overlaps(p, 5, p + 5, 5));
    EXPECT_FALSE(litepgo::overlaps(p + 5, 5, p, 5));
    EXPECT_FALSE(litepgo::overlaps(p, 0, p, 5));
    EXPECT_FALSE(litepgo::overlaps(p, 5, p, 0));
}

} // namespace litepgo::test
