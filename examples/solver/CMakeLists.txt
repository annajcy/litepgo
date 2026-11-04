foreach(example IN ITEMS newton_chain gauss_newton_fit thread_limit)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE litepgo::solver litepgo::warnings)
endforeach()
