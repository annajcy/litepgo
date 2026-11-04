add_executable(custom_response custom_response.cpp)
target_link_libraries(custom_response PRIVATE litepgo::solver litepgo::warnings)

add_executable(mesh_response mesh_response.cpp)
target_link_libraries(mesh_response PRIVATE litepgo::solver litepgo::fem litepgo::warnings)

add_executable(custom_mesh_response custom_mesh_response.cpp)
target_link_libraries(custom_mesh_response PRIVATE litepgo::solver litepgo::warnings)
