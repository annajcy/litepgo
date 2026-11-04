add_executable(energy_composition energy_composition.cpp)
target_link_libraries(energy_composition PRIVATE litepgo::energy litepgo::warnings)

add_executable(custom_energy custom_energy.cpp)
target_link_libraries(custom_energy PRIVATE litepgo::energy litepgo::warnings)

add_executable(mesh_energy_subclass mesh_energy_subclass.cpp)
target_link_libraries(mesh_energy_subclass PRIVATE litepgo::solver litepgo::warnings)
