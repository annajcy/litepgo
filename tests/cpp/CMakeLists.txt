# Test files mirror cpp/litepgo: tests/cpp/<dir>/<name>_gtest.cpp covers cpp/litepgo/<dir>/.
add_executable(litepgo_tests
    core/build_info_gtest.cpp
    core/memory_gtest.cpp
    core/types_gtest.cpp
    linalg/dof_reduction_gtest.cpp
    linalg/reduced_matrix_gtest.cpp
    linalg/sparse_solver_gtest.cpp)
target_link_libraries(litepgo_tests PRIVATE litepgo::litepgo litepgo::warnings GTest::gtest_main)

gtest_discover_tests(litepgo_tests DISCOVERY_MODE PRE_TEST)
