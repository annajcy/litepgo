add_executable(spatial_query spatial_query.cpp)
target_link_libraries(spatial_query PRIVATE litepgo::spatial litepgo::warnings)
