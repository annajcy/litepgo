# Test files mirror cpp/litepgo: tests/cpp/<dir>/<name>_gtest.cpp covers cpp/litepgo/<dir>/.
add_executable(litepgo_tests
    core/build_info_gtest.cpp
    core/busy_scope_gtest.cpp
    core/memory_gtest.cpp
    core/types_gtest.cpp
    energy/affine_mapped_energy_gtest.cpp
    energy/dynamic_pattern_energy_gtest.cpp
    energy/energy_gtest.cpp
    energy/fixed_pattern_energy_gtest.cpp
    energy/quadratic_energy_gtest.cpp
    energy/weighted_energy_sum_gtest.cpp
    fem/basis/basis_gtest.cpp
    fem/dof/element_dof_map_gtest.cpp
    fem/dof/constant_dof_map_gtest.cpp
    fem/dof/element_dof_data_gtest.cpp
    fem/dof/constant_dof_data_gtest.cpp
    fem/field/field_gtest.cpp
    fem/field/local_field_gtest.cpp
    fem/material/material_fields_gtest.cpp
    fem/quadrature/gauss_legendre_1d_gtest.cpp
    fem/basis/hex_q1_basis_gtest.cpp
    fem/field/field_interpolation_gtest.cpp
    fem/basis/lagrange_basis_gtest.cpp
    fem/local/local_kernel_gtest.cpp
    fem/local/dof_derivatives_gtest.cpp
    fem/material/material_inputs_gtest.cpp
    fem/material/multiplicative_kinematics_gtest.cpp
    fem/material/neo_hookean_gtest.cpp
    fem/local/neo_hookean_local_energy_gtest.cpp
    fem/quadrature/quadrature_gtest.cpp
    fem/basis/tensor_product_basis_gtest.cpp
    fem/quadrature/tensor_product_quadrature_gtest.cpp
    fem/dof/vertex_dof_data_gtest.cpp
    fem/dof/vertex_dof_map_gtest.cpp
    fem/local/volume_integration_gtest.cpp
    fem/geometry/volume_quadrature_cache_gtest.cpp
    fem/geometry/volume_reference_geometry_gtest.cpp
    linalg/dof_reduction_gtest.cpp
    linalg/reduced_matrix_gtest.cpp
    linalg/sparse_pattern_gtest.cpp
    linalg/sparse_solver_gtest.cpp
    mesh/mesh_topology_gtest.cpp
    solver/newton_gtest.cpp)
target_link_libraries(litepgo_tests PRIVATE litepgo::litepgo litepgo::warnings GTest::gtest_main)

gtest_discover_tests(litepgo_tests DISCOVERY_MODE PRE_TEST)
