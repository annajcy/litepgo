# Core library. It is header-only until the first sources land; switch to
# add_library(litepgo STATIC ...) and PUBLIC usage requirements then.
add_library(litepgo INTERFACE)
add_library(litepgo::litepgo ALIAS litepgo)

target_include_directories(litepgo INTERFACE "${PROJECT_SOURCE_DIR}/cpp")
target_compile_features(litepgo INTERFACE cxx_std_20)
target_compile_definitions(litepgo INTERFACE EIGEN_DONT_PARALLELIZE LITEPGO_VERSION="${PROJECT_VERSION}")
target_link_libraries(litepgo INTERFACE Eigen3::Eigen TBB::tbb litepgo::accelerate)

if(LITEPGO_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
