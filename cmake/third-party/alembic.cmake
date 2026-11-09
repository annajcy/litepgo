if(TARGET Alembic::Alembic)
    return()
endif()

include(FetchContent)

block(SCOPE_FOR VARIABLES)
    if(NOT TARGET Imath::Imath)
        set(BUILD_SHARED_LIBS OFF)
        set(PYTHON OFF)
        set(PYBIND11 OFF)
        FetchContent_Declare(imath
            URL https://codeload.github.com/AcademySoftwareFoundation/Imath/tar.gz/refs/tags/v3.2.3
            URL_HASH SHA256=e10c12b3f21f45bf08e09d4215d9c7691368d747beebd840de0b6fefed2df9f8
            SYSTEM
            EXCLUDE_FROM_ALL)
        FetchContent_MakeAvailable(imath)
    endif()

    set(ALEMBIC_SHARED_LIBS OFF)
    set(ALEMBIC_DEBUG_WARNINGS_AS_ERRORS OFF)
    set(USE_BINARIES OFF)
    set(USE_HDF5 OFF)
    set(USE_PYALEMBIC OFF)
    set(USE_TESTS OFF)
    # Keep Alembic's host uname probe from overwriting the toolchain architecture.
    set(CMAKE_UNAME OFF)
    FetchContent_Declare(alembic
        URL https://codeload.github.com/alembic/alembic/tar.gz/refs/tags/1.8.12
        URL_HASH SHA256=b6d916c40446e8c502c84273092ab3d98f3f7f6094f8a2b8203d23e2f1d2a4a0
        SYSTEM
        EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(alembic)
    if(imath_SOURCE_DIR)
        # Alembic includes <Imath/half.h>; Imath's build target adds only src/Imath.
        target_include_directories(Alembic SYSTEM PUBLIC "$<BUILD_INTERFACE:${imath_SOURCE_DIR}/src>")
    endif()
endblock()
