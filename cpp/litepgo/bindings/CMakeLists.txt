nanobind_add_module(_core NB_STATIC
    module.cpp
    contact/broad_phase/broad_phase.cpp
    contact/broad_phase/spatial_hash.cpp
    contact/contact_options.cpp
    contact/ipc_contact_energy.cpp
    contact/ipctk_contact_energy.cpp
    core/build_info.cpp
    dof/constant_dof_map.cpp
    dof/dof_reduction.cpp
    dof/element_dof_map.cpp
    dof/stencil_dof_map.cpp
    dof/surface_kinematics.cpp
    dof/vertex_displacement_map.cpp
    dof/vertex_dof_map.cpp
    energy/response/dynamic_pattern_response.cpp
    energy/response/fixed_pattern_response.cpp
    energy/response/mesh_response.cpp
    energy/response/response.cpp
    energy/scalar/affine_mapped_energy.cpp
    energy/scalar/dynamic_pattern_energy.cpp
    energy/scalar/energy.cpp
    energy/scalar/fixed_pattern_energy.cpp
    energy/scalar/mesh_energy.cpp
    energy/scalar/quadratic_energy.cpp
    energy/scalar/vertex_pulling_energy.cpp
    energy/scalar/weighted_energy_sum.cpp
    fem/basis/2d/quad_hermite_basis.cpp
    fem/basis/2d/triangle_p1_basis.cpp
    fem/basis/3d/hex_hermite_basis.cpp
    fem/basis/3d/hex_q1_basis.cpp
    fem/basis/3d/tet_p1_basis.cpp
    fem/element_locator.cpp
    fem/field/deformation_gradient_field.cpp
    fem/field/field.cpp
    fem/field/identity_field.cpp
    fem/field/interpolated_field.cpp
    fem/kinematics/basis_dof_transform.cpp
    fem/kinematics/positive_det_feasible_step.cpp
    fem/kinematics/triangle_mid_edge_kinematics.cpp
    fem/kinematics/volumetric_element.cpp
    fem/local/dof_derivatives.cpp
    fem/local/local_energy.cpp
    fem/local/local_response.cpp
    fem/material/shell/koiter_stvk.cpp
    fem/material/shell/shell_material_local_energy.cpp
    fem/material/shell/shell_material_force_local_response.cpp
    fem/material/shell/shell_fp_force_local_response.cpp
    fem/material/volumetric/neo_hookean.cpp
    fem/material/volumetric/volumetric_fp_force_local_response.cpp
    fem/material/volumetric/volumetric_material_force_local_response.cpp
    fem/material/volumetric/volumetric_material_local_energy.cpp
    fem/point_displacement_map.cpp
    fem/quadrature/2d/gauss_legendre_2d.cpp
    fem/quadrature/2d/triangle_centroid_quadrature.cpp
    fem/quadrature/3d/gauss_legendre_3d.cpp
    fem/quadrature/3d/tet_centroid_quadrature.cpp
    fem/sample_boundary.cpp
    fem/shell_fem_energy.cpp
    fem/shell_fem_force_material_response.cpp
    fem/shell_fem_force_fp_response.cpp
    fem/volumetric_fem_energy.cpp
    fem/volumetric_fem_force_fp_response.cpp
    fem/volumetric_fem_force_material_response.cpp
    linalg/sparse_solver/accelerate_ldlt.cpp
    linalg/sparse_solver/accelerate_llt.cpp
    linalg/sparse_solver/auto_sparse_solver.cpp
    linalg/sparse_solver/eigen_ldlt.cpp
    linalg/sparse_solver/eigen_sparse_lu.cpp
    linalg/sparse_solver/pardiso_ldlt.cpp
    linalg/sparse_solver/pardiso_llt.cpp
    linalg/sparse_solver/sparse_solver.cpp
    mesh/extract_surface.cpp
    mesh/io/msh.cpp
    mesh/io/obj.cpp
    mesh/io/veg.cpp
    mesh/mesh_data.cpp
    mesh/mesh_topology.cpp
    mesh/triangle_mid_edge_stencils.cpp
    solver/gauss_newton.cpp
    solver/newton.cpp
    solver/static_equilibrium_response.cpp
    spatial/spatial_hash.cpp
)
target_link_libraries(_core PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)

# The build tree holds an importable package: _core is built into it, and
# litepgo_dev links the Python sources beside it and registers it with the
# virtual environment through a .pth file. TBB is found through the build rpath.
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/litepgo")

add_custom_target(litepgo_dev ALL
    COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/litepgo_dev.py"
            --source "${PROJECT_SOURCE_DIR}/python/litepgo" --core "$<TARGET_FILE:_core>"
    DEPENDS _core
    COMMENT "Registering this litepgo build with ${Python_EXECUTABLE}"
    VERBATIM)

if(LITEPGO_WITH_GMSH)
    target_link_libraries(_core PRIVATE litepgo::io_gmsh)
endif()
if(LITEPGO_WITH_IPC_TOOLKIT)
    target_link_libraries(_core PRIVATE litepgo::ipctk)
endif()
