add_library(litepgo_spatial INTERFACE)
add_library(litepgo::spatial ALIAS litepgo_spatial)
target_link_libraries(litepgo_spatial INTERFACE litepgo::core)
