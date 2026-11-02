set(TINYOBJLOADER_USE_DOUBLE ON CACHE BOOL "Use double precision OBJ coordinates" FORCE)
set(TINYOBJLOADER_BUILD_TEST_LOADER OFF CACHE BOOL "Disable OBJ example" FORCE)
FetchContent_Declare(tinyobjloader
    URL https://github.com/tinyobjloader/tinyobjloader/archive/refs/tags/v2.0.0rc13.tar.gz
    URL_HASH SHA256=0feb92b838f8ce4aa6eb0ccc32dff30cb64a891e0ec3bde837fca49c78d44334
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(tinyobjloader)
