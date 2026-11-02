# IPC Toolkit is fetched at a pinned release. Its recipes reuse litepgo's
# Eigen3::Eigen and TBB::tbb targets, so the process keeps one Eigen and one
# TBB runtime. The other dependencies it pulls through CPM (libigl,
# Tight-Inclusion, scalable-ccd, spdlog, TinyAD, abseil, robin-map, filib) are
# pinned by tag or commit in its recipes and cached in LITEPGO_CPM_CACHE_DIR,
# which every build tree shares. Set CPM_SOURCE_CACHE to use another cache.
if(NOT DEFINED CPM_SOURCE_CACHE AND NOT DEFINED ENV{CPM_SOURCE_CACHE})
    set(LITEPGO_CPM_CACHE_DIR "${PROJECT_SOURCE_DIR}/build/_cache/cpm" CACHE PATH
        "Download cache for IPC Toolkit's dependencies, shared by every build tree")
    set(CPM_SOURCE_CACHE "${LITEPGO_CPM_CACHE_DIR}" CACHE PATH "" FORCE)
endif()

set(IPC_TOOLKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_WITH_CUDA OFF CACHE BOOL "" FORCE)
# With SIMD, IPC Toolkit adds EIGEN_DONT_VECTORIZE to its public definitions,
# which would compile Eigen differently in litepgo's contact sources than in
# the rest of litepgo.
set(IPC_TOOLKIT_WITH_SIMD OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_WITH_PROFILER OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_WITH_TRACY OFF CACHE BOOL "" FORCE)
set(IPC_TOOLKIT_WITH_CCACHE OFF CACHE BOOL "" FORCE)

set(LITEPGO_IPC_TOOLKIT_VERSION 1.6.1)
FetchContent_Declare(
    ipc_toolkit
    URL https://codeload.github.com/ipc-sim/ipc-toolkit/tar.gz/refs/tags/v${LITEPGO_IPC_TOOLKIT_VERSION}
    URL_HASH SHA256=8ab1fc040324bab951ff721e8639ac2df000ba855688fe97e4bd267b8658f972
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(ipc_toolkit)
