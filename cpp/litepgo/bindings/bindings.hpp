#pragma once
#include <nanobind/nanobind.h>

namespace litepgo::spatial::binding {
void bind_spatial_hash(nanobind::module_& m);
}

namespace litepgo::mesh::binding {
void bind_mesh_mesh_topology(nanobind::module_& m);
void bind_mesh_mesh_data(nanobind::module_& m);
void bind_extract_surface(nanobind::module_& m);
void bind_triangle_mid_edge_stencils(nanobind::module_& m);
void bind_obj(nanobind::module_& m);
void bind_veg(nanobind::module_& m);
void bind_msh(nanobind::module_& m);
} // namespace litepgo::mesh::binding

namespace litepgo::dof::binding {
void bind_dof_reduction(nanobind::module_& m);
void bind_vertex_displacement_map(nanobind::module_& m);
void bind_surface_kinematics(nanobind::module_& m);
void bind_vertex_dof_map(nanobind::module_& m);
void bind_element_dof_map(nanobind::module_& m);
void bind_constant_dof_map(nanobind::module_& m);
void bind_stencil_dof_map(nanobind::module_& m);
} // namespace litepgo::dof::binding

namespace litepgo::linalg::binding {
void bind_sparse_solver(nanobind::module_& m);
void bind_eigen_ldlt(nanobind::module_& m);
void bind_eigen_sparse_lu(nanobind::module_& m);
void bind_accelerate_llt(nanobind::module_& m);
void bind_accelerate_ldlt(nanobind::module_& m);
void bind_pardiso_llt(nanobind::module_& m);
void bind_pardiso_ldlt(nanobind::module_& m);
void bind_auto_sparse_solver(nanobind::module_& m);
} // namespace litepgo::linalg::binding

namespace litepgo::energy::binding {
void bind_energy(nanobind::module_& m);
void bind_fixed_pattern_energy(nanobind::module_& m);
void bind_dynamic_pattern_energy(nanobind::module_& m);
void bind_mesh_energy(nanobind::module_& m);
void bind_quadratic_energy(nanobind::module_& m);
void bind_vertex_pulling_energy(nanobind::module_& m);
void bind_weighted_energy_sum(nanobind::module_& m);
void bind_affine_mapped_energy(nanobind::module_& m);
void bind_response(nanobind::module_& m);
void bind_fixed_pattern_response(nanobind::module_& m);
void bind_dynamic_pattern_response(nanobind::module_& m);
void bind_energy_response_mesh_response(nanobind::module_& m);
} // namespace litepgo::energy::binding

namespace litepgo::fem::binding {
void bind_triangle_p1_basis(nanobind::module_& m);
void bind_quad_hermite_basis(nanobind::module_& m);
void bind_tet_p1_basis(nanobind::module_& m);
void bind_hex_q1_basis(nanobind::module_& m);
void bind_hex_hermite_basis(nanobind::module_& m);
void bind_triangle_centroid_quadrature(nanobind::module_& m);
void bind_tet_centroid_quadrature(nanobind::module_& m);
void bind_fem_quadrature_2d_gauss_legendre_2d(nanobind::module_& m);
void bind_fem_quadrature_3d_gauss_legendre_3d(nanobind::module_& m);
void bind_fem_field_field(nanobind::module_& m);
void bind_identity_field(nanobind::module_& m);
void bind_fem_field_interpolated_field(nanobind::module_& m);
void bind_deformation_gradient_field(nanobind::module_& m);
void bind_fem_material_volumetric_neo_hookean(nanobind::module_& m);
void bind_fem_material_shell_koiter_stvk(nanobind::module_& m);
void bind_triangle_mid_edge_kinematics(nanobind::module_& m);
void bind_fem_element_locator(nanobind::module_& m);
void bind_sample_boundary(nanobind::module_& m);
void bind_fem_local_local_energy(nanobind::module_& m);
void bind_fem_material_volumetric_volumetric_material_local_energy(nanobind::module_& m);
void bind_fem_local_local_response(nanobind::module_& m);
void bind_volumetric_material_force_local_response(nanobind::module_& m);
void bind_volumetric_fp_force_local_response(nanobind::module_& m);
void bind_fem_material_shell_shell_material_local_energy(nanobind::module_& m);
void bind_fem_material_shell_shell_material_force_local_response(nanobind::module_& m);
void bind_fem_material_shell_shell_fp_force_local_response(nanobind::module_& m);
void bind_fem_kinematics_basis_dof_transform(nanobind::module_& m);
void bind_fem_kinematics_volumetric_element(nanobind::module_& m);
void bind_positive_det_feasible_step(nanobind::module_& m);
void bind_dof_derivatives(nanobind::module_& m);
void bind_fem_point_displacement_map(nanobind::module_& m);
void bind_fem_volumetric_fem_energy(nanobind::module_& m);
void bind_fem_volumetric_fem_force_material_response(nanobind::module_& m);
void bind_fem_volumetric_fem_force_fp_response(nanobind::module_& m);
void bind_fem_shell_fem_energy(nanobind::module_& m);
void bind_fem_shell_fem_force_material_response(nanobind::module_& m);
void bind_fem_shell_fem_force_fp_response(nanobind::module_& m);
} // namespace litepgo::fem::binding

namespace litepgo::solver::binding {
void bind_gauss_newton(nanobind::module_& m);
void bind_newton(nanobind::module_& m);
} // namespace litepgo::solver::binding

namespace litepgo::contact::binding {
void bind_contact_contact_options(nanobind::module_& m);
void bind_broad_phase(nanobind::module_& m);
void bind_spatial_hash(nanobind::module_& m);
void bind_ipc_contact_energy(nanobind::module_& m);
void bind_ipctk_contact_energy(nanobind::module_& m);
} // namespace litepgo::contact::binding

namespace litepgo::core::binding {
void bind_build_info(nanobind::module_& m);
}
