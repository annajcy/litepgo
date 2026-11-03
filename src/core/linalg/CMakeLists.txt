add_library(litepgo_linalg STATIC
    sparse_pattern.cpp
    sparse_solver/eigen_ldlt.cpp
    sparse_solver/eigen_sparse_lu.cpp
    sparse_solver/sparse_solver.cpp
)
add_library(litepgo::linalg ALIAS litepgo_linalg)
target_link_libraries(litepgo_linalg
    PUBLIC litepgo::common
    PRIVATE litepgo::warnings)
if(LITEPGO_WITH_ACCELERATE)
    add_library(litepgo_linalg_accelerate STATIC sparse_solver/accelerate_llt.cpp sparse_solver/accelerate_ldlt.cpp)
    add_library(litepgo::linalg_accelerate ALIAS litepgo_linalg_accelerate)
    target_link_libraries(litepgo_linalg_accelerate PUBLIC litepgo::linalg PRIVATE litepgo::accelerate litepgo::warnings)
    target_compile_definitions(litepgo_linalg_accelerate PUBLIC LITEPGO_HAS_ACCELERATE)
endif()
if(LITEPGO_WITH_MKL)
    add_library(litepgo_linalg_mkl STATIC sparse_solver/pardiso_llt.cpp sparse_solver/pardiso_ldlt.cpp)
    add_library(litepgo::linalg_mkl ALIAS litepgo_linalg_mkl)
    target_link_libraries(litepgo_linalg_mkl PUBLIC litepgo::linalg PRIVATE litepgo::mkl litepgo::warnings)
    target_compile_definitions(litepgo_linalg_mkl PUBLIC LITEPGO_HAS_MKL)
endif()
