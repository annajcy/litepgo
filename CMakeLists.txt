cmake_minimum_required(VERSION 3.28)

# FetchContent populates directly instead of through a sub-build per
# dependency (CMake 3.30+), which makes repeat configures much faster.
if(POLICY CMP0168)
    cmake_policy(SET CMP0168 NEW)
endif()

project(litepgo VERSION 0.0.1 LANGUAGES CXX)

option(LITEPGO_BUILD_TESTS "Build the GoogleTest suite" ${PROJECT_IS_TOP_LEVEL})
option(LITEPGO_BUILD_PYTHON "Build the nanobind extension module" OFF)
if(APPLE)
    option(LITEPGO_WITH_ACCELERATE "Use Apple Accelerate for sparse factorization" ON)
else()
    set(LITEPGO_WITH_ACCELERATE OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Point the root compile_commands.json (ignored by git) at this build for clangd.
option(LITEPGO_LINK_COMPILE_COMMANDS "Link compile_commands.json at the source root to this build" ${PROJECT_IS_TOP_LEVEL})
if(LITEPGO_LINK_COMPILE_COMMANDS)
    file(CREATE_LINK "${CMAKE_BINARY_DIR}/compile_commands.json" "${CMAKE_SOURCE_DIR}/compile_commands.json" SYMBOLIC)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/litepgo_warnings.cmake)
include(cmake/litepgo_dependencies.cmake)

add_subdirectory(cpp/litepgo)
add_subdirectory(example/fem EXCLUDE_FROM_ALL)

if(LITEPGO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
    if(LITEPGO_BUILD_PYTHON)
        # PYTHONPATH makes pytest import this build even when the venv's .pth
        # points at another preset.
        add_test(NAME pytest COMMAND "${Python_EXECUTABLE}" -m pytest -q "${PROJECT_SOURCE_DIR}/tests/python"
                 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        set_tests_properties(pytest PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")
    endif()
endif()
