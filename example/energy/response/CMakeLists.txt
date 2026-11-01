add_executable(custom_response custom_response.cpp)
target_link_libraries(custom_response PRIVATE litepgo::litepgo litepgo::warnings)

add_executable(mesh_response mesh_response.cpp)
target_link_libraries(mesh_response PRIVATE litepgo::litepgo litepgo::warnings)
