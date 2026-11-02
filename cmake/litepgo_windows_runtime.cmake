# Windows loads TBB beside each executable/extension, without a caller-specific PATH.
add_custom_target(litepgo_runtime)
add_dependencies(litepgo_runtime TBB::tbb)

function(litepgo_copy_windows_runtime directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type ${target} TYPE)
        if(type STREQUAL "EXECUTABLE" OR type STREQUAL "MODULE_LIBRARY")
            add_dependencies(${target} litepgo_runtime)
            add_custom_command(TARGET litepgo_runtime POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:${target}>"
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:TBB::tbb>" "$<TARGET_FILE_DIR:${target}>"
                VERBATIM)
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${child}")
        if(relative MATCHES "^(src|examples|tests)/")
            litepgo_copy_windows_runtime("${child}")
        endif()
    endforeach()
endfunction()

litepgo_copy_windows_runtime("${PROJECT_SOURCE_DIR}")
