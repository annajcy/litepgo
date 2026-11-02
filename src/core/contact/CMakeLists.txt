add_library(litepgo_contact STATIC
    broad_phase/spatial_hash.cpp
    detail/distance_derivatives.cpp
    detail/ipc_derivatives/edge_edge_mollifier.cpp
    detail/ipc_derivatives/line_line.cpp
    detail/ipc_derivatives/point_line.cpp
    detail/ipc_derivatives/point_plane.cpp
    detail/ipc_derivatives/point_point.cpp
    ipc_contact_energy.cpp
)
add_library(litepgo::contact ALIAS litepgo_contact)
target_link_libraries(litepgo_contact
    PUBLIC litepgo::energy litepgo::mesh litepgo::spatial litepgo::common
    PRIVATE litepgo::warnings)
