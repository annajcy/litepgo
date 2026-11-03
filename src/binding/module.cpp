#include "bindings.hpp"
namespace nb = nanobind;

NB_MODULE(_core, m) {
    m.doc() = "litepgo native extension";
    litepgo::common::binding::bind_build_info(m);

    // Register bases and template tags before their dependent types.
    nb::module_ spatial = m.def_submodule("spatial");
    litepgo::spatial::binding::bind_spatial_hash(spatial);

    nb::module_ mesh = m.def_submodule("mesh");
    mesh.doc() = "Mesh connectivity, geometry, and file I/O (src/core/mesh)";
    litepgo::mesh::binding::bind_mesh_mesh_topology(mesh);
    litepgo::mesh::binding::bind_mesh_mesh_data(mesh);
    litepgo::mesh::binding::bind_extract_surface(mesh);
    litepgo::mesh::binding::bind_triangle_mid_edge_stencils(mesh);
    litepgo::mesh::binding::bind_obj(mesh);
    litepgo::mesh::binding::bind_veg(mesh);
    litepgo::mesh::binding::bind_msh(mesh);
    litepgo::mesh::binding::bind_self_intersections(mesh);

    nb::module_ dof = m.def_submodule("dof");
    dof.doc() = "Degree-of-freedom maps and reductions (src/core/dof)";
    litepgo::dof::binding::bind_dof_reduction(dof);
    litepgo::dof::binding::bind_vertex_displacement_map(dof);
    litepgo::dof::binding::bind_surface_kinematics(dof);
    litepgo::dof::binding::bind_vertex_dof_map(dof);
    litepgo::dof::binding::bind_element_dof_map(dof);
    litepgo::dof::binding::bind_constant_dof_map(dof);
    litepgo::dof::binding::bind_stencil_dof_map(dof);

    nb::module_ linalg = m.def_submodule("linalg");
    linalg.doc() = "Sparse linear solvers (src/core/linalg)";
    litepgo::linalg::binding::bind_sparse_solver(linalg);
    litepgo::linalg::binding::bind_eigen_ldlt(linalg);
    litepgo::linalg::binding::bind_eigen_sparse_lu(linalg);
    litepgo::linalg::binding::bind_accelerate_llt(linalg);
    litepgo::linalg::binding::bind_accelerate_ldlt(linalg);
    litepgo::linalg::binding::bind_pardiso_llt(linalg);
    litepgo::linalg::binding::bind_pardiso_ldlt(linalg);
    litepgo::linalg::binding::bind_auto_sparse_solver(linalg);

    nb::module_ energy = m.def_submodule("energy");
    energy.doc() = "Energies and their composites (src/core/energy)";
    litepgo::energy::binding::bind_energy(energy);
    litepgo::energy::binding::bind_fixed_pattern_energy(energy);
    litepgo::energy::binding::bind_dynamic_pattern_energy(energy);
    litepgo::energy::binding::bind_mesh_energy(energy);
    litepgo::energy::binding::bind_quadratic_energy(energy);
    litepgo::energy::binding::bind_vertex_pulling_energy(energy);
    litepgo::energy::binding::bind_weighted_energy_sum(energy);
    litepgo::energy::binding::bind_affine_mapped_energy(energy);
    litepgo::energy::binding::bind_response(energy);
    litepgo::energy::binding::bind_fixed_pattern_response(energy);
    litepgo::energy::binding::bind_dynamic_pattern_response(energy);
    litepgo::energy::binding::bind_energy_response_mesh_response(energy);

    nb::module_ fem = m.def_submodule("fem");
    fem.doc() = "Finite elements (src/core/fem). Template instantiations are in _templates.";
    litepgo::fem::binding::bind_triangle_p1_basis(fem);
    litepgo::fem::binding::bind_quad_hermite_basis(fem);
    litepgo::fem::binding::bind_tet_p1_basis(fem);
    litepgo::fem::binding::bind_hex_q1_basis(fem);
    litepgo::fem::binding::bind_hex_hermite_basis(fem);
    litepgo::fem::binding::bind_triangle_centroid_quadrature(fem);
    litepgo::fem::binding::bind_tet_centroid_quadrature(fem);
    litepgo::fem::binding::bind_fem_quadrature_2d_gauss_legendre_2d(fem);
    litepgo::fem::binding::bind_fem_quadrature_3d_gauss_legendre_3d(fem);
    litepgo::fem::binding::bind_fem_field_field(fem);
    litepgo::fem::binding::bind_identity_field(fem);
    litepgo::fem::binding::bind_fem_field_interpolated_field(fem);
    litepgo::fem::binding::bind_deformation_gradient_field(fem);
    litepgo::fem::binding::bind_fem_material_volumetric_neo_hookean(fem);
    litepgo::fem::binding::bind_fem_material_shell_koiter_stvk(fem);
    litepgo::fem::binding::bind_triangle_mid_edge_kinematics(fem);
    litepgo::fem::binding::bind_fem_element_locator(fem);
    litepgo::fem::binding::bind_sample_boundary(fem);
    litepgo::fem::binding::bind_fem_local_local_energy(fem);
    litepgo::fem::binding::bind_fem_material_volumetric_volumetric_material_local_energy(fem);
    litepgo::fem::binding::bind_fem_local_local_response(fem);
    litepgo::fem::binding::bind_volumetric_material_force_local_response(fem);
    litepgo::fem::binding::bind_volumetric_fp_force_local_response(fem);
    litepgo::fem::binding::bind_fem_material_shell_shell_material_local_energy(fem);
    litepgo::fem::binding::bind_fem_material_shell_shell_material_force_local_response(fem);
    litepgo::fem::binding::bind_fem_material_shell_shell_fp_force_local_response(fem);
    litepgo::fem::binding::bind_fem_kinematics_basis_dof_transform(fem);
    litepgo::fem::binding::bind_fem_kinematics_volumetric_element(fem);
    litepgo::fem::binding::bind_positive_det_feasible_step(fem);
    litepgo::fem::binding::bind_dof_derivatives(fem);
    litepgo::fem::binding::bind_fem_point_displacement_map(fem);
    litepgo::fem::binding::bind_fem_volumetric_fem_energy(fem);
    litepgo::fem::binding::bind_fem_volumetric_fem_force_material_response(fem);
    litepgo::fem::binding::bind_fem_volumetric_fem_force_fp_response(fem);
    litepgo::fem::binding::bind_fem_shell_fem_energy(fem);
    litepgo::fem::binding::bind_fem_shell_fem_force_material_response(fem);
    litepgo::fem::binding::bind_fem_shell_fem_force_fp_response(fem);

    nb::module_ solver = m.def_submodule("solver");
    solver.doc() = "Nonlinear solvers (src/core/solver)";
    litepgo::solver::binding::bind_gauss_newton(solver);
    litepgo::solver::binding::bind_newton(solver);
    litepgo::solver::binding::bind_static_equilibrium_response(solver);

    nb::module_ contact = m.def_submodule("contact");
    contact.doc() = "Native IPC and optional IPC Toolkit contact (src/core/contact)";
    litepgo::contact::binding::bind_contact_contact_options(contact);
    litepgo::contact::binding::bind_broad_phase(contact);
    litepgo::contact::binding::bind_spatial_hash(contact);
    litepgo::contact::binding::bind_ipc_contact_energy(contact);
    litepgo::contact::binding::bind_ipctk_contact_energy(contact);
}
