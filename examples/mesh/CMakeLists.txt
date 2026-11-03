add_executable(laplacian_optimization laplacian_optimization.cpp)
target_link_libraries(laplacian_optimization PRIVATE litepgo::solver litepgo::mesh litepgo::warnings)

add_executable(arap arap.cpp)
target_link_libraries(arap PRIVATE litepgo::solver litepgo::mesh litepgo::warnings)

if(TARGET litepgo::mesh_cgal)
    add_executable(constrained_remesh constrained_remesh.cpp)
    target_link_libraries(constrained_remesh PRIVATE litepgo::mesh_cgal litepgo::warnings)
endif()
