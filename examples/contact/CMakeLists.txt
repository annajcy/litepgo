add_executable(tet_p1_box_stack tet_p1_box_stack.cpp)
target_link_libraries(tet_p1_box_stack PRIVATE litepgo::contact litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_compile_definitions(tet_p1_box_stack PRIVATE
    LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
    LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")

add_executable(contact_area_weighting contact_area_weighting.cpp)
target_link_libraries(contact_area_weighting PRIVATE litepgo::contact litepgo::warnings)
target_compile_definitions(contact_area_weighting PRIVATE LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")

add_executable(contact_snapshot contact_snapshot.cpp)
target_link_libraries(contact_snapshot PRIVATE litepgo::contact litepgo::warnings)
target_compile_definitions(contact_snapshot PRIVATE LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")

add_executable(contact_pair_weights contact_pair_weights.cpp)
target_link_libraries(contact_pair_weights PRIVATE litepgo::contact litepgo::warnings)
target_compile_definitions(contact_pair_weights PRIVATE LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")

if(LITEPGO_WITH_IPC_TOOLKIT)
    add_executable(contact_benchmark tet_p1_box_stack.cpp)
    target_link_libraries(contact_benchmark PRIVATE litepgo::contact_ipctk litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
    target_compile_definitions(contact_benchmark PRIVATE
        LITEPGO_CONTACT_BENCHMARK=1
        LITEPGO_BENCHMARK_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
        LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
endif()

add_executable(tet_p1_plate_compression tet_p1_plate_compression.cpp)
target_link_libraries(tet_p1_plate_compression PRIVATE litepgo::contact litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_compile_definitions(tet_p1_plate_compression PRIVATE
    LITEPGO_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
    LITEPGO_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output")
