file(GLOB files *.cu *.cuh)

set(LIB_NAME "scheduler_")

add_library(${LIB_NAME} SHARED
    ${files}
    bindings/scheduler_ops.cu
    bindings/bindings.cpp
)

target_include_directories(${LIB_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CUTLASS_DIR}/include
    ${repo-mscclpp_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${LIB_NAME} PUBLIC
    quant
    communication_
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
)
# target_link_directories(${LIB_NAME} PUBLIC ${TORCH_INSTALL_PREFIX}/lib)
# target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LIB_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/eps/scheduler
    CUDA_SEPARABLE_COMPILATION ON
)
