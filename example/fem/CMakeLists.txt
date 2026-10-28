add_executable(hex_q1_mesh_energy hex_q1_mesh_energy.cpp)
target_link_libraries(hex_q1_mesh_energy PRIVATE litepgo::litepgo litepgo::warnings)

add_executable(tet_p1_mesh_energy tet_p1_mesh_energy.cpp)
target_link_libraries(tet_p1_mesh_energy PRIVATE litepgo::litepgo litepgo::warnings)

add_executable(hex_hermite_mesh_energy hex_hermite_mesh_energy.cpp)
target_link_libraries(hex_hermite_mesh_energy PRIVATE litepgo::litepgo litepgo::warnings)

add_executable(shell_stvk_mesh_energy shell_stvk_mesh_energy.cpp)
target_link_libraries(shell_stvk_mesh_energy PRIVATE litepgo::litepgo litepgo::warnings)
