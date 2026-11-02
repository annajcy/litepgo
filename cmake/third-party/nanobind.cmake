# nanobind is fetched from its PyPI sdist, which bundles the robin_map
# submodule that GitHub's tag archive lacks. Its headers then stay in the build
# tree, where clangd can find them after uv's isolated build env is gone.
FetchContent_Declare(
    nanobind
    URL https://files.pythonhosted.org/packages/c7/1f/e28fae715401e8f57400e72aeaff4fde12046d31828164f34f61f0816da8/nanobind-3.1.0.tar.gz
    URL_HASH SHA256=6661e3dc1434d78781ccf7913442d90361dab395c3424efadb63bbb6ddea7ea3
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nanobind)
