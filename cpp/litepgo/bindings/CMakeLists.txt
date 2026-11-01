nanobind_add_module(_core NB_STATIC module.cpp spatial.cpp contact.cpp mesh.cpp dof.cpp linalg.cpp energy.cpp fem.cpp fem_local.cpp fem_energy.cpp solver.cpp response.cpp)
target_link_libraries(_core PRIVATE litepgo::io litepgo::litepgo litepgo::warnings)

# The build tree holds an importable package: _core is built into it, and
# litepgo_dev links the Python sources beside it and registers it with the
# virtual environment through a .pth file. TBB is found through the build rpath.
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/litepgo")

add_custom_target(litepgo_dev ALL
    COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/litepgo_dev.py"
            --source "${PROJECT_SOURCE_DIR}/python/litepgo" --core "$<TARGET_FILE:_core>"
    DEPENDS _core
    COMMENT "Registering this litepgo build with ${Python_EXECUTABLE}"
    VERBATIM)

if(LITEPGO_WITH_GMSH)
    target_link_libraries(_core PRIVATE litepgo::io_gmsh)
endif()
if(LITEPGO_WITH_IPC_TOOLKIT)
    target_link_libraries(_core PRIVATE litepgo::ipctk)
endif()
