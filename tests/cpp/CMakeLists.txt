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
    fem/basis_gtest.cpp
    fem/gauss_legendre_1d_gtest.cpp
    fem/hex_q1_basis_gtest.cpp
    fem/interpolated_field_gtest.cpp
    fem/lagrange_basis_gtest.cpp
    fem/material_types_gtest.cpp
    fem/quadrature_gtest.cpp
    fem/tensor_product_basis_gtest.cpp
    fem/tensor_product_quadrature_gtest.cpp
    fem/vertex_dof_data_gtest.cpp
    fem/vertex_element_dof_map_gtest.cpp
    fem/volume_quadrature_cache_gtest.cpp
    fem/volume_reference_geometry_gtest.cpp
    fem/volumetric_material_gtest.cpp
    linalg/dof_reduction_gtest.cpp
    linalg/reduced_matrix_gtest.cpp
    linalg/sparse_pattern_gtest.cpp
    linalg/sparse_solver_gtest.cpp
    mesh/mesh_topology_gtest.cpp
    solver/newton_gtest.cpp)
target_link_libraries(litepgo_tests PRIVATE litepgo::litepgo litepgo::warnings GTest::gtest_main)

gtest_discover_tests(litepgo_tests DISCOVERY_MODE PRE_TEST)
