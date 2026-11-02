# Windows loads native DLLs beside each executable/extension without a custom PATH.
add_custom_target(litepgo_runtime)
add_dependencies(litepgo_runtime TBB::tbb)

function(litepgo_copy_windows_runtime directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type ${target} TYPE)
        if(type STREQUAL "EXECUTABLE" OR type STREQUAL "MODULE_LIBRARY")
            add_dependencies(${target} litepgo_runtime)
            add_custom_command(TARGET litepgo_runtime POST_BUILD
                COMMAND "${CMAKE_COMMAND}"
                    "-Dlibraries=$<TARGET_FILE:TBB::tbb>;${LITEPGO_MKL_RUNTIME_DLLS}"
                    "-Ddestination=$<TARGET_FILE_DIR:${target}>"
                    -P "${PROJECT_SOURCE_DIR}/cmake/litepgo_copy_runtime.cmake"
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
