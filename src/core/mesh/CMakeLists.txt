add_library(litepgo_mesh STATIC
    extract_surface.cpp
    triangle_mid_edge_stencils.cpp
    io/obj.cpp
    io/veg.cpp
)
add_library(litepgo::mesh ALIAS litepgo_mesh)
target_link_libraries(litepgo_mesh
    PUBLIC litepgo::common
    PRIVATE litepgo::warnings)
include("${PROJECT_SOURCE_DIR}/cmake/third-party/tinyobjloader.cmake")
target_link_libraries(litepgo_mesh PRIVATE tinyobjloader_double)

option(LITEPGO_WITH_GMSH "Enable Gmsh mesh I/O" OFF)
if(LITEPGO_WITH_GMSH)
    include("${PROJECT_SOURCE_DIR}/cmake/third-party/gmsh.cmake")
    add_library(litepgo_mesh_gmsh STATIC io/msh.cpp)
    add_library(litepgo::mesh_gmsh ALIAS litepgo_mesh_gmsh)
    target_link_libraries(litepgo_mesh_gmsh PUBLIC litepgo::mesh PRIVATE Gmsh::Gmsh litepgo::warnings)
    target_compile_definitions(litepgo_mesh_gmsh PUBLIC LITEPGO_HAS_GMSH=1)
endif()

if(LITEPGO_WITH_CGAL)
    add_library(litepgo_mesh_cgal STATIC detail/cgal_surface_mesh.cpp)
    add_library(litepgo::mesh_cgal ALIAS litepgo_mesh_cgal)
    target_link_libraries(litepgo_mesh_cgal PUBLIC litepgo::mesh PRIVATE CGAL::CGAL litepgo::warnings)
    target_compile_definitions(litepgo_mesh_cgal PUBLIC LITEPGO_HAS_CGAL=1)
endif()
