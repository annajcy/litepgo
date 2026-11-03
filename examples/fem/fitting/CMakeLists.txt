add_executable(volume_equilibrium_material_fit volume_equilibrium_material_fit.cpp)
target_link_libraries(volume_equilibrium_material_fit PRIVATE litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_compile_definitions(volume_equilibrium_material_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

add_executable(shell_equilibrium_material_fit shell_equilibrium_material_fit.cpp)
target_link_libraries(shell_equilibrium_material_fit PRIVATE litepgo::solver litepgo::fem litepgo::warnings)

add_executable(volume_equilibrium_fp_fit volume_equilibrium_fp_fit.cpp)
target_link_libraries(volume_equilibrium_fp_fit PRIVATE litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_compile_definitions(volume_equilibrium_fp_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

add_executable(shell_equilibrium_fp_fit shell_equilibrium_fp_fit.cpp)
target_link_libraries(shell_equilibrium_fp_fit PRIVATE litepgo::solver litepgo::fem litepgo::warnings)

add_executable(volume_fixed_pose_fp_fit volume_fixed_pose_fp_fit.cpp)
target_link_libraries(volume_fixed_pose_fp_fit PRIVATE litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_compile_definitions(volume_fixed_pose_fp_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

add_executable(shell_fixed_pose_fp_fit shell_fixed_pose_fp_fit.cpp)
target_link_libraries(shell_fixed_pose_fp_fit PRIVATE litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_compile_definitions(shell_fixed_pose_fp_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

add_executable(shell_fixed_pose_material_fit shell_fixed_pose_material_fit.cpp)
target_link_libraries(shell_fixed_pose_material_fit PRIVATE litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_compile_definitions(shell_fixed_pose_material_fit PRIVATE LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")
