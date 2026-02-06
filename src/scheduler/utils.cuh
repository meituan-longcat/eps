#pragma once

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>  // For std::malloc and std::free
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include <cuda_runtime.h>  // For cudaMalloc and cudaFree

namespace eps {

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define SLEEP_SECONDS(SEC) std::this_thread::sleep_for(std::chrono::milliseconds(unsigned(SEC * 1000)))

// for debug. create type for template
#define CREATE_EMPTY_STRUCT(name) \
  struct name {}

// Default name struct
struct __default_name {};

// Helper function to convert multi-dimensional array to string
template <typename T>
__host__ std::string arrayToString(const T* data, const size_t* shape, size_t numDims, size_t currentDim, size_t& index) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape[currentDim]; ++i) {
    if (currentDim == numDims - 1) {
      oss << data[index++];
    } else {
      oss << arrayToString(data, shape, numDims, currentDim + 1, index);
    }
    if (i < shape[currentDim] - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

// Template declaration for TensorView with a dimension parameter
template <typename T, size_t DIMS, typename NAME = __default_name>
class TensorView;

// Specialization for 1D tensor
template <typename T, typename NAME>
class TensorView<T, 1, NAME> {
 public:
  T* data_;
  size_t dim0_;

  __host__ __device__ TensorView(T* data, size_t dim0) : data_(data), dim0_(dim0) {}

  __host__ __device__ T& at(int i) {
    assert(i >= 0);
    assert(i < dim0_);
    return data_[i];
  }

  __host__ std::string toString() const {
    std::stringstream ss;
    ss << "shape[0]=" << dim0_ << "\n";
    size_t index = 0;
    size_t shape[] = {dim0_};
    return ss.str() + arrayToString(data_, shape, 1, 0, index);
  }

  __host__ __device__ size_t totalNumElems() const { return dim0_; }

  __host__ __device__ size_t totalBytes() const { return totalNumElems() * sizeof(T); }
};

// Specialization for 2D tensor
template <typename T, typename NAME>
class TensorView<T, 2, NAME> {
 public:
  T* data_;
  size_t dim0_, dim1_;

  __host__ __device__ TensorView(T* data, size_t dim0, size_t dim1) : data_(data), dim0_(dim0), dim1_(dim1) {}

  __host__ __device__ T& at(int i, int j) {
    assert(i >= 0);
    assert(i < dim0_);
    assert(j >= 0);
    assert(j < dim1_);
    return data_[i * dim1_ + j];
  }

  __host__ std::string toString() const {
    std::stringstream ss;
    ss << "shape[0]=" << dim0_ << ", shape[1]=" << dim1_ << "\n";
    size_t index = 0;
    size_t shape[] = {dim0_, dim1_};
    return ss.str() + arrayToString(data_, shape, 2, 0, index);
  }

  __host__ __device__ size_t totalNumElems() const { return dim0_ * dim1_; }

  __host__ __device__ size_t totalBytes() const { return totalNumElems() * sizeof(T); }
};

// Specialization for 3D tensor
template <typename T, typename NAME>
class TensorView<T, 3, NAME> {
 public:
  T* data_;
  size_t dim0_, dim1_, dim2_;

  __host__ __device__ TensorView(T* data, size_t dim0, size_t dim1, size_t dim2) : data_(data), dim0_(dim0), dim1_(dim1), dim2_(dim2) {}

  __host__ __device__ T& at(int i, int j, int k) {
    assert(i >= 0);
    assert(i < dim0_);
    assert(j >= 0);
    assert(j < dim1_);
    assert(k >= 0);
    assert(k < dim2_);
    return data_[(i * dim1_ + j) * dim2_ + k];
  }

  __host__ std::string toString() const {
    std::stringstream ss;
    ss << "shape[0]=" << dim0_ << ", shape[1]=" << dim1_ << ", shape[2]=" << dim2_ << "\n";
    size_t index = 0;
    size_t shape[] = {dim0_, dim1_, dim2_};
    return ss.str() + arrayToString(data_, shape, 3, 0, index);
  }

  __host__ __device__ size_t totalNumElems() const { return dim0_ * dim1_ * dim2_; }

  __host__ __device__ size_t totalBytes() const { return totalNumElems() * sizeof(T); }
};

// Helper function for memory allocation and initialization
template <typename T>
__host__ T* mallocAndSet(size_t total_elems, bool is_host, int init_value) {
  T* data_ptr;
  if (is_host) {
    data_ptr = static_cast<T*>(std::malloc(total_elems * sizeof(T)));
    if (!data_ptr) throw std::bad_alloc();
    std::fill(data_ptr, data_ptr + total_elems, init_value);
  } else {
    cudaMalloc(&data_ptr, total_elems * sizeof(T));
    cudaMemset(data_ptr, init_value, total_elems * sizeof(T));  // Assuming init_value is 0 for simplicity
  }
  return data_ptr;
}

// Template declaration for TensorWithData with a dimension parameter
template <typename T, size_t DIMS, typename NAME = __default_name>
class TensorWithData;

// Specialization for 1D TensorWithData
template <typename T, typename NAME>
class TensorWithData<T, 1, NAME> {
 public:
  std::unique_ptr<T, void (*)(T*)> raw_data_ptr_{nullptr, nullptr};
  std::unique_ptr<TensorView<T, 1, NAME>> tensor_view_;

  __host__ static void hostDeleter(T* ptr) { std::free(ptr); }

  __host__ static void deviceDeleter(T* ptr) { cudaFree(ptr); }

  TensorWithData() = delete;
  __host__ TensorWithData(size_t dim0, bool is_host, T init_value) {
    size_t total_elems = dim0;
    T* data_ptr = mallocAndSet<T>(total_elems, is_host, init_value);
    if (is_host) {
      raw_data_ptr_ = std::unique_ptr<T, void (*)(T*)>(data_ptr, hostDeleter);
    } else {
      raw_data_ptr_ = std::unique_ptr<T, void (*)(T*)>(data_ptr, deviceDeleter);
    }
    tensor_view_ = std::make_unique<TensorView<T, 1, NAME>>(raw_data_ptr_.get(), dim0);
  }

  __host__ __device__ T& at(int i) { return tensor_view_->at(i); }

  __host__ std::string toString() const { return tensor_view_->toString(); }

  __host__ __device__ size_t totalNumElems() const { return tensor_view_->totalNumElems(); }

  __host__ __device__ size_t totalBytes() const { return tensor_view_->totalBytes(); }
};

// Specialization for 2D TensorWithData
template <typename T, typename NAME>
class TensorWithData<T, 2, NAME> {
 public:
  std::unique_ptr<T, void (*)(T*)> raw_data_ptr_{nullptr, nullptr};
  std::unique_ptr<TensorView<T, 2, NAME>> tensor_view_;

  __host__ static void hostDeleter(T* ptr) { std::free(ptr); }

  __host__ static void deviceDeleter(T* ptr) { cudaFree(ptr); }

  TensorWithData() = delete;
  __host__ TensorWithData(size_t dim0, size_t dim1, bool is_host, T init_value) {
    size_t total_elems = dim0 * dim1;
    T* data_ptr = mallocAndSet<T>(total_elems, is_host, init_value);
    if (is_host) {
      raw_data_ptr_ = std::unique_ptr<T, void (*)(T*)>(data_ptr, hostDeleter);
    } else {
      raw_data_ptr_ = std::unique_ptr<T, void (*)(T*)>(data_ptr, deviceDeleter);
    }
    tensor_view_ = std::make_unique<TensorView<T, 2, NAME>>(raw_data_ptr_.get(), dim0, dim1);
  }

  __host__ __device__ T& at(int i, int j) { return tensor_view_->at(i, j); }

  __host__ std::string toString() const { return tensor_view_->toString(); }

  __host__ __device__ size_t totalNumElems() const { return tensor_view_->totalNumElems(); }

  __host__ __device__ size_t totalBytes() const { return tensor_view_->totalBytes(); }
};

// Specialization for 3D TensorWithData
template <typename T, typename NAME>
class TensorWithData<T, 3, NAME> {
 public:
  std::unique_ptr<T, void (*)(T*)> raw_data_ptr_{nullptr, nullptr};
  std::unique_ptr<TensorView<T, 3, NAME>> tensor_view_;

  __host__ static void hostDeleter(T* ptr) { std::free(ptr); }

  __host__ static void deviceDeleter(T* ptr) { cudaFree(ptr); }

  TensorWithData() = delete;
  __host__ TensorWithData(size_t dim0, size_t dim1, size_t dim2, bool is_host, T init_value) {
    size_t total_elems = dim0 * dim1 * dim2;
    T* data_ptr = mallocAndSet<T>(total_elems, is_host, init_value);
    if (is_host) {
      raw_data_ptr_ = std::unique_ptr<T, void (*)(T*)>(data_ptr, hostDeleter);
    } else {
      raw_data_ptr_ = std::unique_ptr<T, void (*)(T*)>(data_ptr, deviceDeleter);
    }
    tensor_view_ = std::make_unique<TensorView<T, 3, NAME>>(raw_data_ptr_.get(), dim0, dim1, dim2);
  }

  __host__ __device__ T& at(int i, int j, int k) { return tensor_view_->at(i, j, k); }

  __host__ std::string toString() const { return tensor_view_->toString(); }

  __host__ __device__ size_t totalNumElems() const { return tensor_view_->totalNumElems(); }

  __host__ __device__ size_t totalBytes() const { return tensor_view_->totalBytes(); }
};

/*
// ========misc========
static int getDevice() {
  static int rank = -1;
  if (rank != -1) {
    return rank;
  }
  cudaGetDevice(&rank);
  return rank;
}

// Define the LOG macro
#define LOG(fmt, ...)                                               \
  do {                                                              \
    printf("[rank=%d][%s:%d] : ", getDevice(), __FILE__, __LINE__); \
    printf(fmt, ##__VA_ARGS__);                                     \
    printf("\n");                                                   \
  } while (0)

#define LOG_TENSOR(TYPE, NUM_DIMS, source_var_name, ...)                                                                                      \
  do {                                                                                                                                        \
    TensorWithData<TYPE, NUM_DIMS> source_var_name##_tensor(__VA_ARGS__, true, 0);                                                            \
    cudaMemcpy(source_var_name##_tensor.raw_data_ptr_.get(), source_var_name, source_var_name##_tensor.totalBytes(), cudaMemcpyDeviceToHost); \
    LOG(#source_var_name "_tensor=%s", source_var_name##_tensor.toString().c_str());                                                          \
  } while (0)
*/

#define HIDDEN_SIZE_SWITCH(HIDDEN_SIZE_, ...)                                               \
[&] {                                                                                       \
  if (HIDDEN_SIZE_ == 5120)                                                                 \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 5120;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 2304)                                                                 \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 2304;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 2880)                                                                 \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 2880;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 4096)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 4096;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 1536)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 1536;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 2048)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 2048;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 2560)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 2560;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 3072)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 3072;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 5120)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 5120;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 6144)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 6144;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 7168)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 7168;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 8192)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 8192;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (HIDDEN_SIZE_ == 10240)                                                            \
  {                                                                                         \
    constexpr static int HIDDEN_SIZE = 10240;                                                \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else                                                                                      \
  {                                                                                         \
    throw std::runtime_error("Not supported hidden size: " + std::to_string(HIDDEN_SIZE_)); \
  }                                                                                         \
}

#define EMBED_DIM_SWITCH(EMBED_DIM_, ...)                                                   \
[&] {                                                                                       \
  if (EMBED_DIM_ == 512)                                                                    \
  {                                                                                         \
    constexpr static int EMBED_DIM = 512;                                                   \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else if (EMBED_DIM_ == 256)                                                               \
  {                                                                                         \
    constexpr static int EMBED_DIM = 256;                                                   \
    return __VA_ARGS__();                                                                   \
  }                                                                                         \
  else                                                                                      \
  {                                                                                         \
    throw std::runtime_error("Not supported embed dim: " + std::to_string(EMBED_DIM_));     \
  }                                                                                         \
}

#define COLS_SWITCH(COLS_, ...)                                               \
[&] {                                                                         \
  if (COLS_ == 56)                                                            \
  {                                                                           \
    constexpr static int COLS = 56;                                           \
    return __VA_ARGS__();                                                     \
  }                                                                           \
  else                                                                        \
  {                                                                           \
    throw std::runtime_error("Not supported cols: " + std::to_string(COLS_)); \
  }                                                                           \
}

#define BLOCK_SIZE_SWITCH(BLOCK_SIZE_, ...)                                               \
[&] {                                                                                     \
  if (BLOCK_SIZE_ == 1024)                                                                \
  {                                                                                       \
    constexpr static int BLOCK_SIZE = 1024;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (BLOCK_SIZE_ == 512)                                                            \
  {                                                                                       \
    constexpr static int BLOCK_SIZE = 512;                                                \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (BLOCK_SIZE_ == 256)                                                            \
  {                                                                                       \
    constexpr static int BLOCK_SIZE = 256;                                                \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (BLOCK_SIZE_ == 128)                                                            \
  {                                                                                       \
    constexpr static int BLOCK_SIZE = 128;                                                \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (BLOCK_SIZE_ == 64)                                                            \
  {                                                                                       \
    constexpr static int BLOCK_SIZE = 64;                                                \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else                                                                                    \
  {                                                                                       \
    throw std::runtime_error("Not supported block size: " + std::to_string(BLOCK_SIZE_)); \
  }                                                                                       \
}

#define NUM_EXPERTS_SWITCH(NUM_EXPERTS_, ...)                                               \
[&] {                                                                                     \
  if (NUM_EXPERTS_ == 1952)                                                               \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 1952;                                              \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 980)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 980;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 512)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 512;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 448)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 448;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 256)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 256;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 384)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 384;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 160)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 160;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 128)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 128;                                               \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 64)                                                            \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 64;                                                \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 32)                                                            \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 32;                                                \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 112)                                                           \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 112;                                               \
  else if (NUM_EXPERTS_ == 16)                                                            \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 16;                                                \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else if (NUM_EXPERTS_ == 8)                                                             \
  {                                                                                       \
    constexpr static int NUM_EXPERTS = 8;                                                 \
    return __VA_ARGS__();                                                                 \
  }                                                                                       \
  else                                                                                    \
  {                                                                                       \
    throw std::runtime_error("Not supported num experts: " + std::to_string(NUM_EXPERTS_)); \
  }                                                                                       \
}

}  // namespace eps
