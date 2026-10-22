add_library(litepgo STATIC
    linalg/sparse_solver.cpp
    linalg/eigen_ldlt.cpp
    linalg/eigen_sparse_lu.cpp)
add_library(litepgo::litepgo ALIAS litepgo)

if(LITEPGO_WITH_ACCELERATE)
    target_sources(litepgo PRIVATE linalg/accelerate_llt.cpp linalg/accelerate_ldlt.cpp)
endif()

target_include_directories(litepgo PUBLIC "${PROJECT_SOURCE_DIR}/cpp")
target_compile_features(litepgo PUBLIC cxx_std_20)
target_compile_definitions(litepgo PUBLIC EIGEN_DONT_PARALLELIZE LITEPGO_VERSION="${PROJECT_VERSION}")
target_link_libraries(litepgo PUBLIC Eigen3::Eigen TBB::tbb litepgo::accelerate PRIVATE litepgo::warnings)

if(LITEPGO_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
