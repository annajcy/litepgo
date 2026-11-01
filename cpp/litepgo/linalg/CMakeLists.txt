add_library(litepgo_linalg STATIC
    sparse_pattern.cpp
    sparse_solver/eigen_ldlt.cpp
    sparse_solver/eigen_sparse_lu.cpp
    sparse_solver/sparse_solver.cpp
)
add_library(litepgo::linalg ALIAS litepgo_linalg)
target_link_libraries(litepgo_linalg
    PUBLIC litepgo::core
    PRIVATE litepgo::warnings)
target_link_libraries(litepgo_linalg PUBLIC litepgo::accelerate litepgo::mkl)
if(LITEPGO_WITH_ACCELERATE)
    target_sources(litepgo_linalg PRIVATE sparse_solver/accelerate_llt.cpp sparse_solver/accelerate_ldlt.cpp)
endif()
if(LITEPGO_WITH_MKL)
    target_sources(litepgo_linalg PRIVATE sparse_solver/pardiso_llt.cpp sparse_solver/pardiso_ldlt.cpp)
endif()
