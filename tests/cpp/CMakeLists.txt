# Test files mirror cpp/litepgo: tests/cpp/<dir>/<name>_gtest.cpp covers cpp/litepgo/<dir>/.
add_executable(litepgo_tests
    energy/response/static_equilibrium_response_gtest.cpp
    energy/response/mesh_response_gtest.cpp
    solver/gauss_newton_gtest.cpp
    fem/sample_boundary_gtest.cpp
    dof/vertex_displacement_map_gtest.cpp
    fem/point_displacement_map_gtest.cpp
    fem/element_locator_gtest.cpp
    spatial/spatial_hash_gtest.cpp
    dof/energy_assembler_gtest.cpp
    dof/response_assembler_gtest.cpp
    energy/scalar/mesh_energy_gtest.cpp
    ../example/fem/hex_q1_mesh_energy_gtest.cpp
    ../example/fem/hex_hermite_mesh_energy_gtest.cpp
    ../example/fem/tet_p1_mesh_energy_gtest.cpp
    ../example/fem/shell_stvk_mesh_energy_gtest.cpp
    core/build_info_gtest.cpp
    core/busy_scope_gtest.cpp
    core/memory_gtest.cpp
    core/types_gtest.cpp
    energy/scalar/affine_mapped_energy_gtest.cpp
    energy/scalar/dynamic_pattern_energy_gtest.cpp
    energy/scalar/energy_gtest.cpp
    energy/scalar/fixed_pattern_energy_gtest.cpp
    energy/scalar/quadratic_energy_gtest.cpp
    energy/scalar/vertex_pulling_energy_gtest.cpp
    energy/scalar/weighted_energy_sum_gtest.cpp
    fem/basis/basis_gtest.cpp
    fem/basis/1d/cubic_hermite_basis_gtest.cpp
    fem/basis/2d/tensor_product_basis_2d_gtest.cpp
    fem/basis/2d/quad_hermite_basis_gtest.cpp
    fem/basis/2d/triangle_p1_basis_gtest.cpp
    dof/element_dof_map_gtest.cpp
    dof/constant_dof_map_gtest.cpp
    fem/field/field_gtest.cpp
    fem/field/identity_field_gtest.cpp
    fem/field/shared_field_gtest.cpp
    fem/field/deformation_gradient_field_gtest.cpp
    fem/material/volumetric/material_sources_gtest.cpp
    fem/quadrature/1d/gauss_legendre_1d_gtest.cpp
    fem/basis/3d/hex_q1_basis_gtest.cpp
    fem/basis/3d/hex_hermite_basis_gtest.cpp
    fem/basis/3d/tet_p1_basis_gtest.cpp
    fem/field/interpolated_field_gtest.cpp
    fem/field/interpolated_field_2d_gtest.cpp
    fem/kinematics/basis_dof_transform_gtest.cpp
    fem/basis/1d/linear_lagrange_basis_gtest.cpp
    fem/shell_fem_energy_gtest.cpp
    fem/shell_fem_force_material_response_gtest.cpp
    fem/shell_fem_force_fp_response_gtest.cpp
    fem/local/local_energy_gtest.cpp
    fem/local/local_response_gtest.cpp
    fem/volumetric_fem_force_response_gtest.cpp
    fem/volumetric_fem_force_material_response_gtest.cpp
    fem/material/volumetric/volumetric_material_force_local_response_gtest.cpp
    fem/local/dof_derivatives_gtest.cpp
    fem/kinematics/deformation_gradient_kinematics_gtest.cpp
    fem/kinematics/volumetric_element_gtest.cpp
    fem/material/volumetric/volumetric_material_inputs_gtest.cpp
    fem/material/volumetric/multiplicative_kinematics_gtest.cpp
    fem/kinematics/triangle_mid_edge_kinematics_gtest.cpp
    fem/kinematics/positive_det_feasible_step_gtest.cpp
    fem/material/volumetric/neo_hookean_gtest.cpp
    fem/material/shell/koiter_stvk_gtest.cpp
    fem/material/shell/shell_material_local_energy_gtest.cpp
    fem/material/shell/shell_material_force_local_response_gtest.cpp
    fem/material/shell/shell_fp_force_local_response_gtest.cpp
    fem/material/volumetric/volumetric_material_local_energy_gtest.cpp
    fem/quadrature/quadrature_gtest.cpp
    fem/quadrature/2d/tensor_product_quadrature_2d_gtest.cpp
    fem/quadrature/2d/triangle_centroid_quadrature_gtest.cpp
    fem/quadrature/3d/tet_centroid_quadrature_gtest.cpp
    fem/basis/3d/tensor_product_basis_3d_gtest.cpp
    fem/quadrature/3d/tensor_product_quadrature_3d_gtest.cpp
    dof/vertex_dof_map_gtest.cpp
    dof/stencil_dof_map_gtest.cpp
    fem/material/volumetric/volume_integration_gtest.cpp
    fem/local/surface_integration_gtest.cpp
    fem/material/volumetric/tet_p1_integration_gtest.cpp
    dof/dof_reduction_gtest.cpp
    dof/reduced_matrix_gtest.cpp
    dof/surface_kinematics_gtest.cpp
    linalg/sparse_pattern_gtest.cpp
    linalg/sparse_solver/sparse_solver_gtest.cpp
    linalg/sparse_solver/pardiso_threading_gtest.cpp
    mesh/extract_surface_gtest.cpp
    mesh/mesh_topology_gtest.cpp
    mesh/triangle_mid_edge_stencils_gtest.cpp
    solver/newton_gtest.cpp)
target_link_libraries(litepgo_tests PRIVATE litepgo::litepgo litepgo::warnings GTest::gtest_main)

gtest_discover_tests(litepgo_tests DISCOVERY_MODE PRE_TEST)

# Run the standalone examples as loaded static-equilibrium regression tests.
foreach(example tet_p1_mesh_energy hex_q1_mesh_energy hex_hermite_mesh_energy shell_stvk_mesh_energy)
    add_dependencies(litepgo_tests ${example})
    add_test(NAME FemEquilibrium.${example} COMMAND $<TARGET_FILE:${example}>)
endforeach()

# The plastic twist examples solve a released and a clamped bar with a plastic torsion Fp.
foreach(example tet_p1_plastic_twist hex_q1_plastic_twist hex_hermite_plastic_twist)
    add_dependencies(litepgo_tests ${example})
    add_test(NAME FemPlasticTwist.${example} COMMAND $<TARGET_FILE:${example}>)
endforeach()

target_sources(litepgo_tests PRIVATE contact/ipc_contact_energy_gtest.cpp contact/broad_phase/broad_phase_gtest.cpp)
add_dependencies(litepgo_tests tet_p1_box_stack)
add_test(NAME ContactEquilibrium.tet_p1_box_stack COMMAND $<TARGET_FILE:tet_p1_box_stack>)
if(LITEPGO_WITH_IPC_TOOLKIT)
    target_sources(litepgo_tests PRIVATE contact/ipctk_contact_energy_gtest.cpp)
    target_link_libraries(litepgo_tests PRIVATE litepgo::ipctk)
endif()

add_dependencies(litepgo_tests spatial_query point_embedding)
add_test(NAME SpatialQueryExample COMMAND $<TARGET_FILE:spatial_query>)
add_test(NAME PointEmbeddingExample COMMAND $<TARGET_FILE:point_embedding>)

add_dependencies(litepgo_tests material_fit)
add_test(NAME MaterialFit COMMAND $<TARGET_FILE:material_fit>)

add_dependencies(litepgo_tests custom_energy custom_response)
add_test(NAME EnergyApiExample COMMAND $<TARGET_FILE:custom_energy>)
add_test(NAME ResponseApiExample COMMAND $<TARGET_FILE:custom_response>)
add_dependencies(litepgo_tests mesh_response)
add_test(NAME MeshResponseExample COMMAND $<TARGET_FILE:mesh_response>)

add_dependencies(litepgo_tests shell_material_fit)
add_test(NAME ShellMaterialFit COMMAND $<TARGET_FILE:shell_material_fit>)

add_dependencies(litepgo_tests diagonal_fp_fit)
add_test(NAME DiagonalFpFit COMMAND $<TARGET_FILE:diagonal_fp_fit>)

add_dependencies(litepgo_tests shell_fp_fit)
add_test(NAME ShellFpFit COMMAND $<TARGET_FILE:shell_fp_fit>)

add_dependencies(litepgo_tests shell_plastic_field)
add_test(NAME ShellPlasticField COMMAND $<TARGET_FILE:shell_plastic_field> ${CMAKE_CURRENT_BINARY_DIR}/shell_plastic_field)

add_dependencies(litepgo_tests shell_equilibrium_material_fit)
add_test(NAME ShellEquilibriumMaterialFit COMMAND $<TARGET_FILE:shell_equilibrium_material_fit>)
