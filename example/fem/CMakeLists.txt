add_executable(hex_q1_mesh_energy hex_q1_mesh_energy.cpp)
target_link_libraries(hex_q1_mesh_energy PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)

add_executable(tet_p1_mesh_energy tet_p1_mesh_energy.cpp)
target_link_libraries(tet_p1_mesh_energy PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)

add_executable(hex_hermite_mesh_energy hex_hermite_mesh_energy.cpp)
target_link_libraries(hex_hermite_mesh_energy PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)

add_executable(shell_stvk_mesh_energy shell_stvk_mesh_energy.cpp)
target_link_libraries(shell_stvk_mesh_energy PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)

foreach(target hex_q1_mesh_energy tet_p1_mesh_energy shell_stvk_mesh_energy hex_hermite_mesh_energy)
    target_compile_definitions(${target} PRIVATE
        LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
        LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
endforeach()

add_subdirectory(plastic_twist)

add_executable(point_embedding point_embedding.cpp)
target_link_libraries(point_embedding PRIVATE litepgo::litepgo litepgo::warnings)

add_executable(material_fit material_fit.cpp)
target_link_libraries(material_fit PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
target_compile_definitions(material_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

add_executable(shell_material_fit shell_material_fit.cpp)
target_link_libraries(shell_material_fit PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
target_compile_definitions(shell_material_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")
