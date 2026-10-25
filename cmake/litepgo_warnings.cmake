# Warning flags shared by every litepgo target.
add_library(litepgo_warnings INTERFACE)
add_library(litepgo::warnings ALIAS litepgo_warnings)

if(MSVC)
    target_compile_options(litepgo_warnings INTERFACE /W4 /permissive-)
else()
    target_compile_options(litepgo_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()
