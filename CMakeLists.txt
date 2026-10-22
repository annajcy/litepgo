cmake_minimum_required(VERSION 3.28)

project(litepgo VERSION 0.0.1 LANGUAGES CXX)

option(LITEPGO_BUILD_TESTS "Build the GoogleTest suite" ${PROJECT_IS_TOP_LEVEL})
option(LITEPGO_BUILD_PYTHON "Build the nanobind extension module" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/LitepgoWarnings.cmake)
include(cmake/LitepgoDependencies.cmake)

add_subdirectory(cpp/litepgo)

if(LITEPGO_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if(LITEPGO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
