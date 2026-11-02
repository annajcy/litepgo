file(MAKE_DIRECTORY "${destination}")
foreach(library IN LISTS libraries)
    get_filename_component(name "${library}" NAME)
    # Share the large MKL kernels across output directories; copy across volumes.
    file(CREATE_LINK "${library}" "${destination}/${name}" COPY_ON_ERROR RESULT result)
    if(NOT result STREQUAL "0")
        message(FATAL_ERROR "Cannot stage ${library} in ${destination}: ${result}")
    endif()
endforeach()
