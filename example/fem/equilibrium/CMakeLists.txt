foreach(example tet_p1_corner_pull hex_q1_corner_pull hex_hermite_corner_pull shell_saddle_pull)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
    target_compile_definitions(${example} PRIVATE
        LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
        LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
endforeach()
