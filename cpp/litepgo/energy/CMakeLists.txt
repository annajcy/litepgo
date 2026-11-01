add_library(litepgo_energy STATIC
    response/mesh_response.cpp
    response/response.cpp
    scalar/affine_mapped_energy.cpp
    scalar/dynamic_pattern_energy.cpp
    scalar/energy.cpp
    scalar/mesh_energy.cpp
    scalar/quadratic_energy.cpp
    scalar/vertex_pulling_energy.cpp
    scalar/weighted_energy_sum.cpp
)
add_library(litepgo::energy ALIAS litepgo_energy)
target_link_libraries(litepgo_energy
    PUBLIC litepgo::dof litepgo::linalg litepgo::core
    PRIVATE litepgo::warnings)
