# Eigen is fetched at a pinned version. litepgo parallelizes with TBB itself,
# so Eigen's own threading is disabled.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    URL_HASH SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(Eigen3)
