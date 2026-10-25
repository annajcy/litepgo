# Apple Accelerate provides the sparse LDLT backend on macOS through
# <Eigen/AccelerateSupport>. Only the framework is linked; Eigen dense code
# does not route through Accelerate BLAS.
add_library(litepgo_accelerate INTERFACE)
add_library(litepgo::accelerate ALIAS litepgo_accelerate)

if(LITEPGO_WITH_ACCELERATE)
    find_library(LITEPGO_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(litepgo_accelerate INTERFACE "${LITEPGO_ACCELERATE_FRAMEWORK}")
    target_compile_definitions(litepgo_accelerate INTERFACE LITEPGO_HAS_ACCELERATE)
endif()
