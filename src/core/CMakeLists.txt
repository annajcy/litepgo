# Keep public <litepgo/...> includes independent of the source directory layout.
file(GLOB_RECURSE litepgo_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
foreach(header IN LISTS litepgo_headers)
    file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}" "${header}")
    file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/include/litepgo/${relative}"
         CONTENT "#pragma once\n#include \"${header}\"\n")
endforeach()

add_subdirectory(common)
add_subdirectory(spatial)
add_subdirectory(linalg)
add_subdirectory(mesh)
add_subdirectory(dof)
add_subdirectory(energy)
add_subdirectory(fem)
add_subdirectory(solver)
add_subdirectory(contact)

# Native module aggregate; module libraries never depend on it.
add_library(litepgo_core INTERFACE)
add_library(litepgo::core ALIAS litepgo_core)
add_library(litepgo::litepgo ALIAS litepgo_core)
target_link_libraries(litepgo_core INTERFACE
    litepgo::contact litepgo::solver litepgo::fem litepgo::energy litepgo::dof
    litepgo::mesh litepgo::linalg litepgo::spatial litepgo::common)

include("${PROJECT_SOURCE_DIR}/cmake/litepgo_module_graph.cmake")
foreach(module litepgo_core litepgo_mesh_alembic litepgo_mesh_gmsh litepgo_mesh_cgal litepgo_contact_ipctk)
    if(TARGET ${module})
        litepgo_check_module_dag(${module} "")
    endif()
endforeach()
