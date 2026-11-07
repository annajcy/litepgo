# The JIT references symbols after the link, so retain the native module archives.
file(CONFIGURE OUTPUT empty.cpp CONTENT "// Symbols come from the litepgo modules.\n")
add_library(litepgo_tutorial_cpp SHARED "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp")
target_link_libraries(litepgo_tutorial_cpp PRIVATE litepgo::core)
set(modules litepgo_contact litepgo_solver litepgo_fem litepgo_energy litepgo_dof litepgo_mesh litepgo_linalg)
foreach(backend accelerate mkl)
    if(TARGET litepgo_linalg_${backend})
        list(APPEND modules litepgo_linalg_${backend})
    endif()
endforeach()
list(JOIN modules "," archives)
set_property(TARGET litepgo_tutorial_cpp PROPERTY LINK_LIBRARY_OVERRIDE "WHOLE_ARCHIVE,${archives}")

# Pass the same headers and definitions to xeus-cpp that native consumers use.
set(includes "$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:litepgo_tutorial_cpp,INCLUDE_DIRECTORIES>>")
set(definitions "$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:litepgo_tutorial_cpp,COMPILE_DEFINITIONS>>")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/kernel-arguments.txt" CONTENT
    "$<TARGET_FILE:litepgo_tutorial_cpp>\n-I$<JOIN:${includes},\n-I>\n-D$<JOIN:${definitions},\n-D>\n")
