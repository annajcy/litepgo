add_library(litepgo_fem STATIC
    basis/2d/triangle_p1_basis.cpp
    basis/3d/tet_p1_basis.cpp
    kinematics/positive_det_feasible_step.cpp
    kinematics/triangle_mid_edge_kinematics.cpp
    assembly/dof_derivatives.cpp
    material/shell/koiter_stvk.cpp
    material/volumetric/multiplicative_kinematics.cpp
    material/volumetric/neo_hookean.cpp
    embedding/sample_boundary.cpp
)
add_library(litepgo::fem ALIAS litepgo_fem)
target_link_libraries(litepgo_fem
    PUBLIC litepgo::energy litepgo::dof litepgo::mesh litepgo::spatial litepgo::common
    PRIVATE litepgo::warnings)
