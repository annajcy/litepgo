# Third-party dependencies. Versions are pinned so every build sees the same sources.
include(FetchContent)

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
    # scikit-build-core provides nanobind from the build requirements; a manual
    # configure finds it through the active Python environment.
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
endif()
