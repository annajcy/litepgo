add_library(litepgo STATIC
    dof/assembler.cpp
    energy/mesh_energy.cpp
    energy/affine_mapped_energy.cpp
    energy/dynamic_pattern_energy.cpp
    energy/energy.cpp
    energy/quadratic_energy.cpp
    energy/weighted_energy_sum.cpp
    fem/material/multiplicative_kinematics.cpp
    fem/material/neo_hookean.cpp
    fem/local/dof_derivatives.cpp
    fem/local/local_energy.cpp
    fem/basis/3d/hex_q1_basis.cpp
    fem/basis/3d/hex_hermite_basis.cpp
    fem/basis/3d/tet_p1_basis.cpp
    fem/field/hex_hermite_dof_transform.cpp
    dof/dof_reduction.cpp
    linalg/sparse_solver/eigen_ldlt.cpp
    linalg/sparse_solver/eigen_sparse_lu.cpp
    linalg/reduced_matrix.cpp
    linalg/sparse_pattern.cpp
    linalg/sparse_solver/sparse_solver.cpp
    solver/newton.cpp)
add_library(litepgo::litepgo ALIAS litepgo)

if(LITEPGO_WITH_ACCELERATE)
    target_sources(litepgo PRIVATE linalg/sparse_solver/accelerate_llt.cpp linalg/sparse_solver/accelerate_ldlt.cpp)
endif()

target_include_directories(litepgo PUBLIC "${PROJECT_SOURCE_DIR}/cpp")
target_compile_features(litepgo PUBLIC cxx_std_20)
target_compile_definitions(litepgo PUBLIC EIGEN_DONT_PARALLELIZE LITEPGO_VERSION="${PROJECT_VERSION}")
target_link_libraries(litepgo PUBLIC Eigen3::Eigen TBB::tbb litepgo::accelerate PRIVATE litepgo::warnings)

if(LITEPGO_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
