add_library(litepgo_fem STATIC
    basis/2d/quad_hermite_basis.cpp
    basis/2d/triangle_p1_basis.cpp
    basis/3d/hex_hermite_basis.cpp
    basis/3d/hex_q1_basis.cpp
    basis/3d/tet_p1_basis.cpp
    kinematics/positive_det_feasible_step.cpp
    kinematics/triangle_mid_edge_kinematics.cpp
    local/dof_derivatives.cpp
    local/local_energy.cpp
    local/local_response.cpp
    material/shell/koiter_stvk.cpp
    material/volumetric/multiplicative_kinematics.cpp
    material/volumetric/neo_hookean.cpp
    sample_boundary.cpp
)
add_library(litepgo::fem ALIAS litepgo_fem)
target_link_libraries(litepgo_fem
    PUBLIC litepgo::energy litepgo::dof litepgo::mesh litepgo::spatial litepgo::core
    PRIVATE litepgo::warnings)
