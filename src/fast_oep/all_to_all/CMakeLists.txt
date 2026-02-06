# All-to-All library

add_library(fast_oep_all_to_all_lib STATIC
    fast_oep_kernels.cu
    fast_oep.cu
)
target_link_libraries(fast_oep_all_to_all_lib PUBLIC communication_)
target_include_directories(fast_oep_all_to_all_lib PUBLIC ${PROJECT_SOURCE_DIR} ${repo-mscclpp_SOURCE_DIR}/include)
set_cuda_compile_options(fast_oep_all_to_all_lib)
