# === Library targets ===
add_subdirectory(all_to_all)

# Main shared library
add_library(fast_oep SHARED
    bindings/all_to_all_ops.cu
    bindings/bindings.cpp
)
target_link_libraries(fast_oep PUBLIC
    fast_oep_all_to_all_lib
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
)
set_target_properties(fast_oep PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/eps/fast_oep
    CUDA_SEPARABLE_COMPILATION ON
)
