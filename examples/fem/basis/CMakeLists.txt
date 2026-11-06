add_executable(tet_p1_basis tet_p1_basis.cpp)
target_link_libraries(tet_p1_basis PRIVATE litepgo::fem litepgo::warnings)

add_executable(hermite_interpolation hermite_interpolation.cpp)
target_link_libraries(hermite_interpolation PRIVATE litepgo::fem litepgo::warnings)

add_executable(interpolated_response interpolated_response.cpp)
target_link_libraries(interpolated_response PRIVATE litepgo::fem litepgo::warnings)
