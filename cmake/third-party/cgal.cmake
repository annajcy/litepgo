# Use the release archive with a unified Boost header tree.
FetchContent_Declare(boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-b2-nodocs.tar.xz
    URL_HASH SHA256=ea7b982002cc9dfbe59b0b217b206f470dc75f3de0bb2973d844118934d82411
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SOURCE_SUBDIR boost
    OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(boost)
if(NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED)
    target_include_directories(Boost::boost INTERFACE "${boost_SOURCE_DIR}")
endif()

# Use the release's package config without adding its top-level build.
FetchContent_Declare(cgal
    URL https://github.com/CGAL/cgal/releases/download/v6.2.1/CGAL-6.2.1-library.tar.xz
    URL_HASH SHA256=8e7ce1e30e8ffcb3e69d153c3a561bb1f38bbb4becd8e6544cbaa0fb2889af62
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SOURCE_SUBDIR lib/cmake/CGAL)
FetchContent_MakeAvailable(cgal)
set(CGAL_DIR "${cgal_SOURCE_DIR}/lib/cmake/CGAL")
# Boost.Multiprecision provides exact arithmetic without GMP or MPFR.
set(CGAL_DISABLE_GMP ON)
find_package(CGAL 6.2.1 EXACT CONFIG REQUIRED NO_DEFAULT_PATH)
