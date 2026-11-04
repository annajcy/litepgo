add_executable(custom_response custom_response.cpp)
target_link_libraries(custom_response PRIVATE litepgo::solver litepgo::warnings)

add_executable(mesh_response mesh_response.cpp)
target_link_libraries(mesh_response PRIVATE litepgo::solver litepgo::fem litepgo::warnings)

add_executable(mesh_response_subclass mesh_response_subclass.cpp)
target_link_libraries(mesh_response_subclass PRIVATE litepgo::solver litepgo::warnings)
