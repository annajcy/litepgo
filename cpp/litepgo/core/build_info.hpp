#pragma once

#include <Eigen/Version>
#include <oneapi/tbb/version.h>

#include <string>
#include <vector>

namespace litepgo {

struct BuildInfo {
    std::string version;
    std::string eigen_version;
    std::string tbb_header_version;
    std::string tbb_runtime_version;
    std::vector<std::string> sparse_backends;
};

// What this build was compiled against, and the TBB runtime actually loaded.
inline BuildInfo build_info() {
    BuildInfo info;
    info.version = LITEPGO_VERSION;
    // Eigen 5 keeps EIGEN_WORLD/MAJOR/MINOR_VERSION at 3.x for compatibility.
    info.eigen_version = EIGEN_VERSION_STRING;
    info.tbb_header_version = TBB_VERSION_STRING;
    info.tbb_runtime_version = TBB_runtime_version();
    info.sparse_backends = {"EigenLDLT", "EigenSparseLU"};
#if defined(LITEPGO_HAS_ACCELERATE)
    info.sparse_backends.push_back("AccelerateLLT");
    info.sparse_backends.push_back("AccelerateLDLT");
#endif
    return info;
}

} // namespace litepgo
