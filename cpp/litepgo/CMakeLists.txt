# Core library. It is header-only until the first sources land; switch to
# add_library(litepgo STATIC ...) and PUBLIC usage requirements then.
add_library(litepgo INTERFACE)
add_library(litepgo::litepgo ALIAS litepgo)

target_include_directories(litepgo INTERFACE "${PROJECT_SOURCE_DIR}/cpp")
target_compile_features(litepgo INTERFACE cxx_std_20)
