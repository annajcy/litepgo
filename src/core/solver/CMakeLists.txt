add_library(litepgo_solver STATIC
    gauss_newton.cpp
    newton.cpp
    static_equilibrium_response.cpp
)
add_library(litepgo::solver ALIAS litepgo_solver)
target_link_libraries(litepgo_solver
    PUBLIC litepgo::energy litepgo::dof litepgo::linalg litepgo::common
    PRIVATE litepgo::warnings)

foreach(backend accelerate mkl)
    if(TARGET litepgo::linalg_${backend})
        target_link_libraries(litepgo_solver PUBLIC litepgo::linalg_${backend})
    endif()
endforeach()
