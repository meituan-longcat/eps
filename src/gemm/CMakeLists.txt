# Get all Grouped GEMM sources
file(GLOB_RECURSE AOK_GROUPED_SRC_CU aok_grouped_gemm/src/operations/*.cu)

# Fetch CUTLASS repository
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG        eefa171318b79cbe2e78514d4cce5cd0fe919d0c
)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only of header library")
FetchContent_MakeAvailable(cutlass)
# Set CUTLASS_DIR to be fetched content's source directory
set(CUTLASS_DIR ${cutlass_SOURCE_DIR})
message("============AOK uses internal CUTLASS_DIR: ${CUTLASS_DIR}============")

message(
    STATUS
    "AokGroupedGemm srcs ${AOK_GROUPED_SRC_CU}"
)

# Add AokGroupedGemm library
add_library(eps_aok_grouped_gemm STATIC ${AOK_GROUPED_SRC_CU})
set_property(TARGET eps_aok_grouped_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET eps_aok_grouped_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(eps_aok_grouped_gemm PRIVATE ${PROJECT_SOURCE_DIR}/src/gemm/cutlass_extensions ${PROJECT_SOURCE_DIR}/src/gemm)
target_include_directories(eps_aok_grouped_gemm PRIVATE ${PROJECT_SOURCE_DIR} ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include/ ${CUTLASS_DIR}/examples/)
target_include_directories(eps_aok_grouped_gemm PRIVATE ${PROJECT_SOURCE_DIR}/src/gemm/aok_grouped_gemm/include)