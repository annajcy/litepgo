# Third-party dependencies. Fetched sources are pinned so every build sees the same code.
include(FetchContent)

include(cmake/third-party/eigen.cmake)
include(cmake/third-party/tbb.cmake)
include(cmake/third-party/accelerate.cmake)

if(LITEPGO_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
endif()

if(LITEPGO_BUILD_PYTHON)
    include(cmake/litepgo_python.cmake)
    include(cmake/third-party/nanobind.cmake)
endif()
