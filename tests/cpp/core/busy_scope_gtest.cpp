#include "litepgo/core/busy_scope.h"

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>

TEST(BusyScope, RejectsOverlapAndReleasesOnExit) {
    std::atomic<bool> busy = false;
    {
        const litepgo::BusyScope outer(busy, "Thing");
        EXPECT_TRUE(busy.load());
        EXPECT_THROW(litepgo::BusyScope(busy, "Thing"), std::logic_error);
        EXPECT_TRUE(busy.load()); // the failed scope must not release the outer one
    }
    EXPECT_FALSE(busy.load());
    EXPECT_NO_THROW(litepgo::BusyScope(busy, "Thing"));
}
