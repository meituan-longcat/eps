# All-to-All library

add_library(fast_ep_all_to_all_lib STATIC
    fast_ep_kernels.cu
    fast_ep.cu
)
target_link_libraries(fast_ep_all_to_all_lib PUBLIC communication_)
target_include_directories(fast_ep_all_to_all_lib PUBLIC ${PROJECT_SOURCE_DIR} ${repo-mscclpp_SOURCE_DIR}/include)
set_cuda_compile_options(fast_ep_all_to_all_lib)
