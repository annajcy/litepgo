#include "litepgo/core/build_info.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

TEST(BuildInfo, ReportsPinnedDependencies) {
    const litepgo::BuildInfo info = litepgo::build_info();
    EXPECT_EQ(info.version, LITEPGO_VERSION);
    EXPECT_EQ(info.eigen_version, "5.0.1");
    EXPECT_FALSE(info.tbb_header_version.empty());
    // The loaded TBB runtime is the one the headers came from.
    EXPECT_EQ(info.tbb_runtime_version, info.tbb_header_version);
}

TEST(BuildInfo, ListsTheCompiledSparseBackends) {
    std::vector<std::string> expected = {"EigenLDLT", "EigenSparseLU"};
#if defined(LITEPGO_HAS_ACCELERATE)
    expected.push_back("AccelerateLLT");
    expected.push_back("AccelerateLDLT");
#endif
    EXPECT_EQ(litepgo::build_info().sparse_backends, expected);
}
