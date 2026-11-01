add_subdirectory(common)
add_subdirectory(spatial)
add_subdirectory(linalg)
add_subdirectory(mesh)
add_subdirectory(dof)
add_subdirectory(energy)
add_subdirectory(fem)
add_subdirectory(solver)
add_subdirectory(contact)

# Compatibility entry point; module libraries never depend on this aggregate.
add_library(litepgo INTERFACE)
add_library(litepgo::litepgo ALIAS litepgo)
target_link_libraries(litepgo INTERFACE
    litepgo::contact litepgo::solver litepgo::fem litepgo::energy litepgo::dof
    litepgo::mesh litepgo::linalg litepgo::spatial litepgo::common)

add_library(litepgo_io STATIC mesh/io/obj.cpp mesh/io/veg.cpp)
add_library(litepgo::io ALIAS litepgo_io)
target_link_libraries(litepgo_io PUBLIC litepgo::mesh PRIVATE litepgo::warnings)
include("${PROJECT_SOURCE_DIR}/cmake/third-party/tinyobjloader.cmake")
target_link_libraries(litepgo_io PRIVATE tinyobjloader_double)

option(LITEPGO_WITH_GMSH "Enable Gmsh mesh I/O" OFF)
if(LITEPGO_WITH_GMSH)
    include("${PROJECT_SOURCE_DIR}/cmake/third-party/gmsh.cmake")
    add_library(litepgo_io_gmsh STATIC mesh/io/msh.cpp)
    add_library(litepgo::io_gmsh ALIAS litepgo_io_gmsh)
    target_link_libraries(litepgo_io_gmsh PUBLIC litepgo::io PRIVATE Gmsh::Gmsh)
    target_compile_definitions(litepgo_io_gmsh PUBLIC LITEPGO_HAS_GMSH=1)
endif()

if(LITEPGO_WITH_IPC_TOOLKIT)
    add_library(litepgo_ipctk STATIC contact/ipctk_contact_energy.cpp)
    add_library(litepgo::ipctk ALIAS litepgo_ipctk)
    # IPC Toolkit types stay out of litepgo's headers, so consumers never see them.
    target_link_libraries(litepgo_ipctk PUBLIC litepgo::contact PRIVATE ipc::toolkit litepgo::warnings)
    target_compile_definitions(litepgo_ipctk PUBLIC LITEPGO_HAS_IPC_TOOLKIT=1)
endif()

include("${PROJECT_SOURCE_DIR}/cmake/litepgo_module_graph.cmake")
foreach(module litepgo litepgo_io litepgo_io_gmsh litepgo_ipctk)
    if(TARGET ${module})
        litepgo_check_module_dag(${module} "")
    endif()
endforeach()

if(LITEPGO_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
