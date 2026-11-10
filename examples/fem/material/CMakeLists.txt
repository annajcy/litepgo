add_executable(shell_interpolated_fp shell_interpolated_fp.cpp)
target_link_libraries(shell_interpolated_fp PRIVATE litepgo::solver litepgo::fem litepgo::mesh)
target_compile_definitions(shell_interpolated_fp PRIVATE
    LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
    LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
