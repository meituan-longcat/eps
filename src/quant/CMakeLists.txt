file(GLOB files *.cu)

set(LIB_NAME "quant")
add_library(${LIB_NAME} SHARED ${files})
target_include_directories(${LIB_NAME} PRIVATE ${PROJECT_SOURCE_DIR} ${repo-mscclpp_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} mscclpp)
# target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --keep>)
# target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
