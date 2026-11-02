# oneTBB is fetched at a pinned version and built as a shared library, so one
# TBB runtime serves the whole process. The Python install ships it next to
# _core (see src/binding/CMakeLists.txt).
set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB_STRICT OFF CACHE BOOL "" FORCE)
set(TBB_INSTALL OFF CACHE BOOL "" FORCE)
set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)
set(TBB_ENABLE_IPO OFF CACHE BOOL "" FORCE)
# Never build hwloc-dependent tbbbind from whatever the host happens to have.
set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    TBB
    URL https://codeload.github.com/uxlfoundation/oneTBB/tar.gz/refs/tags/v2023.1.0
    URL_HASH SHA256=191288b52e1e6b17198000b64d77d194bb65e791be46ebc606e9b091781e2070
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(TBB)
# oneMKL loads the release-named TBB library even in Debug builds.
if(LITEPGO_WITH_MKL)
    set_target_properties(tbb PROPERTIES DEBUG_POSTFIX "")
    if(WIN32)
        # CMake supplies the correct library; suppress MSVC's tbb12_debug.lib pragma.
        target_compile_definitions(tbb INTERFACE __TBB_NO_IMPLICIT_LINKAGE=1)
    endif()
endif()
