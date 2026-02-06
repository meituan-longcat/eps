# === Library targets ===
add_subdirectory(all_to_all)

# Main shared library
add_library(fast_ep SHARED
    bindings/all_to_all_ops.cu
    bindings/bindings.cpp
)
target_link_libraries(fast_ep PUBLIC
    fast_ep_all_to_all_lib
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
)
set_target_properties(fast_ep PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/eps/fast_ep
    CUDA_SEPARABLE_COMPILATION ON
)
