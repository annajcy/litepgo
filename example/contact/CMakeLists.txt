add_executable(tet_p1_box_stack tet_p1_box_stack.cpp)
target_link_libraries(tet_p1_box_stack PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)
target_compile_definitions(tet_p1_box_stack PRIVATE
    LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
    LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
