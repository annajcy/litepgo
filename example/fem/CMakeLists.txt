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

add_executable(diagonal_fp_fit diagonal_fp_fit.cpp)
target_link_libraries(diagonal_fp_fit PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
target_compile_definitions(diagonal_fp_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

add_executable(shell_fp_fit shell_fp_fit.cpp)
target_link_libraries(shell_fp_fit PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
target_compile_definitions(shell_fp_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

add_executable(shell_plastic_field shell_plastic_field.cpp)
target_link_libraries(shell_plastic_field PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
target_compile_definitions(shell_plastic_field PRIVATE LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")

add_executable(shell_equilibrium_material_fit shell_equilibrium_material_fit.cpp)
target_link_libraries(shell_equilibrium_material_fit PRIVATE litepgo::litepgo litepgo::warnings)
