add_executable(laplacian_optimization laplacian_optimization.cpp)
target_link_libraries(laplacian_optimization PRIVATE litepgo::solver litepgo::io litepgo::warnings)

add_executable(arap arap.cpp)
target_link_libraries(arap PRIVATE litepgo::solver litepgo::io litepgo::warnings)
