add_library(litepgo_mesh STATIC
    extract_surface.cpp
    triangle_mid_edge_stencils.cpp
)
add_library(litepgo::mesh ALIAS litepgo_mesh)
target_link_libraries(litepgo_mesh
    PUBLIC litepgo::common
    PRIVATE litepgo::warnings)
