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

if(LITEPGO_WITH_IPC_TOOLKIT)
    add_library(litepgo_contact_ipctk STATIC ipctk_contact_energy.cpp)
    add_library(litepgo::contact_ipctk ALIAS litepgo_contact_ipctk)
    # IPC Toolkit types stay out of litepgo's headers, so consumers never see them.
    target_link_libraries(litepgo_contact_ipctk PUBLIC litepgo::contact PRIVATE ipc::toolkit litepgo::warnings)
    target_compile_definitions(litepgo_contact_ipctk PUBLIC LITEPGO_HAS_IPC_TOOLKIT=1)
endif()
