add_library(litepgo STATIC
    dof/assembler.cpp
    mesh/extract_surface.cpp
    mesh/triangle_mid_edge_stencils.cpp
    energy/mesh_energy.cpp
    energy/affine_mapped_energy.cpp
    energy/dynamic_pattern_energy.cpp
    energy/energy.cpp
    energy/quadratic_energy.cpp
    energy/vertex_pulling_energy.cpp
    energy/weighted_energy_sum.cpp
    fem/material/volumetric/multiplicative_kinematics.cpp
    fem/material/volumetric/neo_hookean.cpp
    fem/material/shell/koiter_stvk.cpp
    fem/local/dof_derivatives.cpp
    fem/local/local_energy.cpp
    fem/kinematics/triangle_mid_edge_kinematics.cpp
    fem/kinematics/positive_det_feasible_step.cpp
    fem/basis/2d/quad_hermite_basis.cpp
    fem/basis/2d/triangle_p1_basis.cpp
    fem/basis/3d/hex_q1_basis.cpp
    fem/basis/3d/hex_hermite_basis.cpp
    fem/basis/3d/tet_p1_basis.cpp
    dof/dof_reduction.cpp
    dof/reduced_matrix.cpp
    linalg/sparse_solver/eigen_ldlt.cpp
    linalg/sparse_solver/eigen_sparse_lu.cpp
    linalg/sparse_pattern.cpp
    linalg/sparse_solver/sparse_solver.cpp
    solver/newton.cpp)
add_library(litepgo::litepgo ALIAS litepgo)

if(LITEPGO_WITH_ACCELERATE)
    target_sources(litepgo PRIVATE linalg/sparse_solver/accelerate_llt.cpp linalg/sparse_solver/accelerate_ldlt.cpp)
endif()

target_include_directories(litepgo PUBLIC "${PROJECT_SOURCE_DIR}/cpp")
target_compile_features(litepgo PUBLIC cxx_std_20)
target_compile_definitions(litepgo PUBLIC EIGEN_DONT_PARALLELIZE LITEPGO_VERSION="${PROJECT_VERSION}")
target_link_libraries(litepgo PUBLIC Eigen3::Eigen TBB::tbb litepgo::accelerate PRIVATE litepgo::warnings)

add_library(litepgo_io STATIC mesh/io/obj.cpp mesh/io/veg.cpp)
add_library(litepgo::io ALIAS litepgo_io)
target_link_libraries(litepgo_io PUBLIC litepgo::litepgo)
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

if(LITEPGO_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
