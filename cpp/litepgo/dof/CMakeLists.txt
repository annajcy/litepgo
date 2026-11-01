add_library(litepgo_dof STATIC
    dof_reduction.cpp
    energy_assembler.cpp
    reduced_matrix.cpp
    response_assembler.cpp
    surface_kinematics.cpp
)
add_library(litepgo::dof ALIAS litepgo_dof)
target_link_libraries(litepgo_dof
    PUBLIC litepgo::mesh litepgo::linalg litepgo::common
    PRIVATE litepgo::warnings)
