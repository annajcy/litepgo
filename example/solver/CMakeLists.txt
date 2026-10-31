foreach(example IN ITEMS newton_chain gauss_newton_fit)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE litepgo::litepgo litepgo::warnings)
endforeach()
