#include "litepgo/common/busy_scope.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>

namespace litepgo::test {

TEST(BusyScope, RejectsOverlapAndReleasesOnExit) {
    std::atomic<bool> busy = false;
    {
        const litepgo::detail::BusyScope outer(busy, "Thing");
        EXPECT_TRUE(busy.load());
        EXPECT_THROW(litepgo::detail::BusyScope(busy, "Thing"), std::logic_error);
        EXPECT_TRUE(busy.load()); // the failed scope must not release the outer one
    }
    EXPECT_FALSE(busy.load());
    EXPECT_NO_THROW(litepgo::detail::BusyScope(busy, "Thing"));
}

} // namespace litepgo::test
