# Test executables link only the modules exercised by their source files.
add_custom_target(litepgo_tests ALL)
add_dependencies(litepgo_tests thread_limit)
add_test(NAME TbbThreadLimitExample COMMAND thread_limit)

add_executable(litepgo_solver_tests
    solver/static_equilibrium_response_gtest.cpp
    solver/gauss_newton_gtest.cpp
    solver/newton_gtest.cpp
)
target_link_libraries(litepgo_solver_tests PRIVATE
    litepgo::solver litepgo::fem litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_solver_tests)
gtest_discover_tests(litepgo_solver_tests DISCOVERY_MODE PRE_TEST)

add_executable(litepgo_energy_tests
    energy/response/response_hessian_gtest.cpp
    energy/response/composed_response_gtest.cpp
    energy/response/mesh_response_gtest.cpp
    energy/scalar/mesh_energy_gtest.cpp
    energy/scalar/affine_mapped_energy_gtest.cpp
    energy/scalar/dynamic_pattern_energy_gtest.cpp
    energy/scalar/energy_gtest.cpp
    energy/scalar/fixed_pattern_energy_gtest.cpp
    energy/scalar/quadratic_energy_gtest.cpp
    energy/scalar/vertex_pulling_energy_gtest.cpp
    energy/scalar/weighted_energy_sum_gtest.cpp
)
target_link_libraries(litepgo_energy_tests PRIVATE
    litepgo::fem litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_energy_tests)
gtest_discover_tests(litepgo_energy_tests DISCOVERY_MODE PRE_TEST)

add_executable(litepgo_fem_tests
    fem/sample_boundary_gtest.cpp
    fem/point_displacement_map_gtest.cpp
    fem/element_locator_gtest.cpp
    fem/basis/basis_gtest.cpp
    fem/basis/1d/cubic_hermite_basis_gtest.cpp
    fem/basis/2d/quad_hermite_basis_gtest.cpp
    fem/basis/2d/triangle_p1_basis_gtest.cpp
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
    fem/shell_fem_energy_regression_gtest.cpp
    fem/shell_fem_force_material_response_gtest.cpp
    fem/shell_fem_force_fp_response_gtest.cpp
    fem/local/local_energy_gtest.cpp
    fem/local/local_response_gtest.cpp
    fem/volumetric_fem_force_response_gtest.cpp
    fem/volumetric_fem_force_material_response_gtest.cpp
    fem/volumetric_fem_energy_hex_q1_gtest.cpp
    fem/volumetric_fem_energy_hex_hermite_gtest.cpp
    fem/volumetric_fem_energy_tet_p1_gtest.cpp
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
    fem/quadrature/3d/tensor_product_quadrature_3d_gtest.cpp
    fem/material/volumetric/volume_integration_gtest.cpp
    fem/local/surface_integration_gtest.cpp
    fem/material/volumetric/tet_p1_integration_gtest.cpp
)
target_link_libraries(litepgo_fem_tests PRIVATE
    litepgo::solver litepgo::fem litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_fem_tests)
gtest_discover_tests(litepgo_fem_tests DISCOVERY_MODE PRE_TEST)

add_executable(litepgo_dof_tests
    dof/vertex_displacement_map_gtest.cpp
    dof/energy_assembler_gtest.cpp
    dof/response_assembler_gtest.cpp
    dof/element_dof_map_gtest.cpp
    dof/constant_dof_map_gtest.cpp
    dof/vertex_dof_map_gtest.cpp
    dof/stencil_dof_map_gtest.cpp
    dof/dof_reduction_gtest.cpp
    dof/reduced_matrix_gtest.cpp
    dof/surface_kinematics_gtest.cpp
)
target_link_libraries(litepgo_dof_tests PRIVATE
    litepgo::fem litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_dof_tests)
gtest_discover_tests(litepgo_dof_tests DISCOVERY_MODE PRE_TEST)

add_executable(litepgo_spatial_tests
    spatial/spatial_hash_gtest.cpp
)
target_link_libraries(litepgo_spatial_tests PRIVATE
    litepgo::spatial litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_spatial_tests)
gtest_discover_tests(litepgo_spatial_tests DISCOVERY_MODE PRE_TEST)

add_executable(litepgo_common_tests
    common/build_info_gtest.cpp
    common/busy_scope_gtest.cpp
    common/memory_gtest.cpp
    common/types_gtest.cpp
)
target_link_libraries(litepgo_common_tests PRIVATE
    litepgo::common litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_common_tests)
gtest_discover_tests(litepgo_common_tests DISCOVERY_MODE PRE_TEST)

add_executable(litepgo_linalg_tests
    linalg/sparse_pattern_gtest.cpp
    linalg/sparse_solver/sparse_solver_gtest.cpp
    linalg/sparse_solver/pardiso_threading_gtest.cpp
)
target_link_libraries(litepgo_linalg_tests PRIVATE
    litepgo::linalg litepgo::warnings GTest::gtest_main ${CMAKE_DL_LIBS})
foreach(backend accelerate mkl)
    if(TARGET litepgo::linalg_${backend})
        target_link_libraries(litepgo_linalg_tests PRIVATE litepgo::linalg_${backend})
    endif()
endforeach()
add_dependencies(litepgo_tests litepgo_linalg_tests)
gtest_discover_tests(litepgo_linalg_tests DISCOVERY_MODE PRE_TEST)

# Exercise base-only linkage even when platform backends are configured.
add_executable(litepgo_linalg_base_tests linalg/sparse_solver/sparse_solver_gtest.cpp)
target_link_libraries(litepgo_linalg_base_tests PRIVATE litepgo::linalg litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_linalg_base_tests)
gtest_discover_tests(litepgo_linalg_base_tests DISCOVERY_MODE PRE_TEST TEST_PREFIX "LinalgBase.")

add_executable(litepgo_mesh_tests
    mesh/extract_surface_gtest.cpp
    mesh/mesh_topology_gtest.cpp
    mesh/triangle_mid_edge_stencils_gtest.cpp
)
target_link_libraries(litepgo_mesh_tests PRIVATE
    litepgo::mesh litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_mesh_tests)
gtest_discover_tests(litepgo_mesh_tests DISCOVERY_MODE PRE_TEST)

if(TARGET litepgo::mesh_cgal)
    add_executable(litepgo_mesh_cgal_tests
        mesh/cgal_surface_mesh_gtest.cpp
        mesh/isotropic_remesh_gtest.cpp
        mesh/self_intersections_gtest.cpp
    )
    target_link_libraries(litepgo_mesh_cgal_tests PRIVATE
        litepgo::mesh_cgal CGAL::CGAL litepgo::warnings GTest::gtest_main)
    add_dependencies(litepgo_tests litepgo_mesh_cgal_tests)
    gtest_discover_tests(litepgo_mesh_cgal_tests DISCOVERY_MODE PRE_TEST)
endif()

add_executable(litepgo_contact_tests
    contact/contact_pair_weights_gtest.cpp
    contact/ipc_contact_energy_gtest.cpp
    contact/broad_phase/broad_phase_gtest.cpp
)
target_link_libraries(litepgo_contact_tests PRIVATE
    litepgo::contact litepgo::warnings GTest::gtest_main)
add_dependencies(litepgo_tests litepgo_contact_tests)
gtest_discover_tests(litepgo_contact_tests DISCOVERY_MODE PRE_TEST)

# Run the standalone examples as loaded static-equilibrium regression tests.
foreach(example tet_p1_corner_pull hex_q1_corner_pull hex_hermite_corner_pull shell_saddle_pull)
    add_dependencies(litepgo_tests ${example})
    set(equilibrium_input)
    if(example STREQUAL "hex_hermite_corner_pull")
        set(equilibrium_input "${PROJECT_SOURCE_DIR}/tests/assets/box-hex.veg")
    endif()
    add_test(NAME FemEquilibrium.${example} COMMAND $<TARGET_FILE:${example}> ${equilibrium_input})
endforeach()

# The plastic twist examples solve a released and a clamped bar with a plastic torsion Fp.
foreach(example tet_p1_plastic_twist hex_q1_plastic_twist hex_hermite_plastic_twist)
    add_dependencies(litepgo_tests ${example})
    set(plastic_twist_input)
    if(example STREQUAL "hex_hermite_plastic_twist")
        # Keep the high-order regression practical in unoptimized Debug builds.
        set(plastic_twist_input "${PROJECT_SOURCE_DIR}/tests/assets/box-hex.veg")
    endif()
    add_test(NAME FemPlasticTwist.${example} COMMAND $<TARGET_FILE:${example}> ${plastic_twist_input})
endforeach()

add_dependencies(litepgo_tests tet_p1_box_stack)
add_test(NAME ContactEquilibrium.tet_p1_box_stack COMMAND $<TARGET_FILE:tet_p1_box_stack>)
add_dependencies(litepgo_tests contact_area_weighting)
add_test(NAME ContactAreaWeighting COMMAND $<TARGET_FILE:contact_area_weighting>)
add_dependencies(litepgo_tests contact_snapshot)
add_test(NAME ContactSnapshotExample COMMAND $<TARGET_FILE:contact_snapshot>)
add_dependencies(litepgo_tests contact_pair_weights)
add_test(NAME ContactPairWeightsExample COMMAND $<TARGET_FILE:contact_pair_weights>)
if(LITEPGO_WITH_IPC_TOOLKIT)
    target_sources(litepgo_contact_tests PRIVATE contact/ipctk_contact_energy_gtest.cpp)
    target_link_libraries(litepgo_contact_tests PRIVATE litepgo::contact_ipctk)
endif()

add_dependencies(litepgo_tests spatial_query point_embedding)
add_test(NAME SpatialQueryExample COMMAND $<TARGET_FILE:spatial_query>)
add_test(NAME PointEmbeddingExample COMMAND $<TARGET_FILE:point_embedding>)

add_dependencies(litepgo_tests tet_p1_basis)
add_test(NAME FemBasisExample COMMAND $<TARGET_FILE:tet_p1_basis>)
add_dependencies(litepgo_tests hermite_interpolation)
add_test(NAME FemHermiteInterpolationExample COMMAND $<TARGET_FILE:hermite_interpolation>)
add_dependencies(litepgo_tests quadrature_integration)
add_test(NAME FemQuadratureExample COMMAND $<TARGET_FILE:quadrature_integration>)

add_dependencies(litepgo_tests volume_equilibrium_material_fit)
add_test(NAME FemFit.volume_equilibrium_material_fit COMMAND $<TARGET_FILE:volume_equilibrium_material_fit>)

add_dependencies(litepgo_tests custom_energy custom_response)
add_test(NAME EnergyApiExample COMMAND $<TARGET_FILE:custom_energy>)
add_test(NAME ResponseApiExample COMMAND $<TARGET_FILE:custom_response>)
add_dependencies(litepgo_tests mesh_response)
add_test(NAME MeshResponseExample COMMAND $<TARGET_FILE:mesh_response>)

add_dependencies(litepgo_tests shell_fixed_pose_material_fit)
add_test(NAME FemFit.shell_fixed_pose_material_fit COMMAND $<TARGET_FILE:shell_fixed_pose_material_fit>)

add_dependencies(litepgo_tests volume_fixed_pose_fp_fit)
add_test(NAME FemFit.volume_fixed_pose_fp_fit COMMAND $<TARGET_FILE:volume_fixed_pose_fp_fit>)

add_dependencies(litepgo_tests shell_fixed_pose_fp_fit)
add_test(NAME FemFit.shell_fixed_pose_fp_fit COMMAND $<TARGET_FILE:shell_fixed_pose_fp_fit>)

add_dependencies(litepgo_tests shell_plastic_field)
add_test(NAME ShellPlasticField COMMAND $<TARGET_FILE:shell_plastic_field> ${CMAKE_CURRENT_BINARY_DIR}/shell_plastic_field)

add_dependencies(litepgo_tests shell_equilibrium_material_fit)
add_test(NAME FemFit.shell_equilibrium_material_fit COMMAND $<TARGET_FILE:shell_equilibrium_material_fit>)

add_dependencies(litepgo_tests volume_equilibrium_fp_fit shell_equilibrium_fp_fit)
add_test(NAME FemFit.volume_equilibrium_fp_fit COMMAND $<TARGET_FILE:volume_equilibrium_fp_fit>)
add_test(NAME FemFit.shell_equilibrium_fp_fit COMMAND $<TARGET_FILE:shell_equilibrium_fp_fit>)

add_dependencies(litepgo_tests mesh_energy_subclass)
add_test(NAME MeshEnergySubclassExample COMMAND $<TARGET_FILE:mesh_energy_subclass>)

add_dependencies(litepgo_tests mesh_response_subclass)
add_test(NAME MeshResponseSubclassExample COMMAND $<TARGET_FILE:mesh_response_subclass>)

add_dependencies(litepgo_tests response_hessian)
add_test(NAME ResponseHessianExample COMMAND $<TARGET_FILE:response_hessian>)
