add_executable(energy_composition energy_composition.cpp)
target_link_libraries(energy_composition PRIVATE litepgo::litepgo litepgo::warnings)
