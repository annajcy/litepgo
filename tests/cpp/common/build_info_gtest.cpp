#include "litepgo/common/build_info.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace litepgo::test {

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
#if defined(LITEPGO_CONFIGURED_ACCELERATE)
    expected.push_back("AccelerateLLT");
    expected.push_back("AccelerateLDLT");
#endif
#if defined(LITEPGO_CONFIGURED_MKL)
    expected.push_back("PardisoLLT");
    expected.push_back("PardisoLDLT");
#endif
    EXPECT_EQ(litepgo::build_info().sparse_backends, expected);
}

TEST(BuildInfo, ReportsTheContactBackend) {
#if defined(LITEPGO_IPC_TOOLKIT_VERSION)
    EXPECT_EQ(litepgo::build_info().ipc_toolkit_version, "1.6.1");
#else
    EXPECT_TRUE(litepgo::build_info().ipc_toolkit_version.empty());
#endif
}

} // namespace litepgo::test
