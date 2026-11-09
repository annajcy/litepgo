# shortcut: all binding .cpp files belong to this module. Use an explicit list if that changes.
file(GLOB_RECURSE binding_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

nanobind_add_module(litepgo_binding NB_STATIC ${binding_sources})
target_link_libraries(litepgo_binding PRIVATE litepgo::contact litepgo::solver litepgo::fem litepgo::mesh litepgo::warnings)
target_include_directories(litepgo_binding PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# The build tree holds an importable package: _core is built into it, and
# litepgo_python links the Python sources beside it and registers it with the
# virtual environment through a .pth file. TBB uses the build rpath on Unix;
# cmake/litepgo_windows_runtime.cmake copies it beside the extension on Windows.
set_target_properties(litepgo_binding PROPERTIES
    OUTPUT_NAME _core
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/litepgo"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/litepgo")

add_custom_target(litepgo_python ALL
    COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/litepgo_dev.py"
            --source "${PROJECT_SOURCE_DIR}/src/python" --core "$<TARGET_FILE:litepgo_binding>"
    DEPENDS litepgo_binding
    COMMENT "Registering this litepgo build with ${Python_EXECUTABLE}"
    VERBATIM)

if(LITEPGO_WITH_GMSH)
    target_link_libraries(litepgo_binding PRIVATE litepgo::mesh_gmsh)
endif()
if(TARGET litepgo::mesh_alembic)
    target_link_libraries(litepgo_binding PRIVATE litepgo::mesh_alembic)
endif()
if(TARGET litepgo::mesh_cgal)
    target_link_libraries(litepgo_binding PRIVATE litepgo::mesh_cgal)
endif()
if(LITEPGO_WITH_IPC_TOOLKIT)
    target_link_libraries(litepgo_binding PRIVATE litepgo::contact_ipctk)
endif()
