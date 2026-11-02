add_executable(point_embedding point_embedding.cpp)
target_link_libraries(point_embedding PRIVATE litepgo::fem litepgo::warnings)
