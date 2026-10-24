add_library(litepgo STATIC
    energy/affine_mapped_energy.cpp
    energy/dynamic_pattern_energy.cpp
    energy/energy.cpp
    energy/quadratic_energy.cpp
    energy/weighted_energy_sum.cpp
    fem/material/multiplicative_kinematics.cpp
    fem/material/neo_hookean.cpp
    fem/local/neo_hookean_local_energy.cpp
    fem/field/field_interpolation.cpp
    fem/field/deformation_gradient_field.cpp
    fem/field/interpolated_value_field.cpp
    fem/material/material_fields.cpp
    fem/local/dof_derivatives.cpp
    fem/local/local_energy.cpp
    fem/local/local_kernel.cpp
    fem/basis/hex_q1_basis.cpp
    linalg/dof_reduction.cpp
    linalg/eigen_ldlt.cpp
    linalg/eigen_sparse_lu.cpp
    linalg/reduced_matrix.cpp
    linalg/sparse_pattern.cpp
    linalg/sparse_solver.cpp
    solver/newton.cpp)
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
