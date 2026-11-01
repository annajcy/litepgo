foreach(example tet_p1_plastic_twist hex_q1_plastic_twist hex_hermite_plastic_twist shell_plastic_field)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
    target_compile_definitions(${example} PRIVATE
        LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
        LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
endforeach()
