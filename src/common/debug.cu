#include "src/common/debug.cuh"

#include <unistd.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace eps {

std::string get_hostname() {
  char hostname[HOST_NAME_MAX];
  if (gethostname(hostname, HOST_NAME_MAX) == 0) {
    return std::string(hostname);
  } else {
    throw std::runtime_error("Error getting host name");
  }
}

void copy_print(std::string name, const int *d, int n)
{
    cudaDeviceSynchronize();
    std::vector<int> h(n);
    cudaMemcpy(h.data(), d, sizeof(int) * n, cudaMemcpyDeviceToHost);

    std::cout << name << ": " << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cout << h[i] << " ";
    }
    std::cout << std::endl;
}

void copy_print(std::string name, const half *d, int n)
{
    cudaDeviceSynchronize();
    std::vector<half> h(n);
    cudaMemcpy(h.data(), d, sizeof(half) * n, cudaMemcpyDeviceToHost);

    std::cout << name << ": " << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cout << (float)h[i] << " ";
    }
    std::cout << std::endl;
}

void copy_print(std::string name, const float *d, int n)
{
    cudaDeviceSynchronize();
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, sizeof(float) * n, cudaMemcpyDeviceToHost);

    std::cout << name << ": " << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cout << h[i] << " ";
    }
    std::cout << std::endl;
}

const char *_cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorString(error);
}

template <typename... Args>
std::string fmtstr(const std::string& format, Args... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);
}


void syncAndCheckEps(const char *const file, int const line, bool force) {
  static std::string debug_mode = []() {
    const char* level_name = std::getenv("EPS_DEBUG_LEVEL");
    if (level_name == nullptr) {
      return "default";
    } else {
      return level_name;
    }
  }();

  if (force || debug_mode == "DEBUG" || debug_mode == "VERBOSE") {
    int rank;
    cudaGetDevice(&rank);
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
      std::string msg = fmtstr("\033[31m[%s][rank%d][ERROR][%s:%d] CUDA runtime error: %s\033[0m\n", get_hostname().c_str(), rank, file, line, _cudaGetErrorEnum(result));
      throw std::runtime_error(msg);
    } else {
      if (force || debug_mode == "VERBOSE") {
        fprintf(stderr, "[%s][Rank%d] EPS pass %s:%d\n", get_hostname().c_str(), rank, file, line);
      }
    }
  }
}

template <typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr) {
  T* tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
    cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%7.5f ", (float)tmp[ii * stride + jj]);
      } else {
        printf("%7d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

template void printMatrix(float* ptr, int m, int k, int stride, bool is_device_ptr);
template void printMatrix(half* ptr, int m, int k, int stride, bool is_device_ptr);
#ifdef ENABLE_BF16
template void printMatrix(__nv_bfloat16* ptr, int m, int k, int stride, bool is_device_ptr);
#endif
#ifdef ENABLE_FP8
template void printMatrix(__nv_fp8_e4m3* ptr, int m, int k, int stride, bool is_device_ptr);
#endif

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr) {
  typedef unsigned long long T;
  T* tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
    cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%4llu ", tmp[ii * stride + jj]);
      } else {
        printf("%4d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr) {
  typedef int T;
  T* tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
    cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%4d ", tmp[ii * stride + jj]);
      } else {
        printf("%4d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

void printMatrix(int64_t* ptr, int m, int k, int stride, bool is_device_ptr) {
  typedef int64_t T;
  T* tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
    cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%4ld ", tmp[ii * stride + jj]);
      } else {
        printf("%4d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

// multiple definitions for msvc
#ifndef _MSC_VER
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr) {
  typedef size_t T;
  T* tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
    cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%4ld ", tmp[ii * stride + jj]);
      } else {
        printf("%4d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}
#endif

template <typename T>
void print_to_file(
    const T* result,
    const int size,
    const char* file,
    cudaStream_t stream,
    std::ios::openmode open_mode) {
  cudaDeviceSynchronize();
  printf("[INFO] file: %s with size %d.\n", file, size);
  std::ofstream outFile(file, open_mode);
  if (outFile) {
    T* tmp = new T[size];
    cudaMemcpyAsync(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost, stream);
    for (int i = 0; i < size; ++i) {
      float val = (float)(tmp[i]);
      outFile << val << std::endl;
    }
    delete[] tmp;
  } else {
    throw std::runtime_error(std::string("[ERROR] Cannot open file: ") + file + "\n");
  }
  cudaDeviceSynchronize();
}

template void print_to_file(
    const int64_t* result,
    const int size,
    const char* file,
    cudaStream_t stream,
    std::ios::openmode open_mode);
template void
print_to_file(const int* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);
template void
print_to_file(const float* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);
template void
print_to_file(const half* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);
#ifdef ENABLE_BF16
template void print_to_file(
    const __nv_bfloat16* result,
    const int size,
    const char* file,
    cudaStream_t stream,
    std::ios::openmode open_mode);
#endif
#ifdef ENABLE_FP8
template void
print_to_file(const __nv_fp8_e4m3* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);
#endif


// ============getMatrixStr Starts============
template <typename T>
std::string getMatrixStr(T* ptr, int m, int k, int stride, bool is_device_ptr) {
  std::stringstream ss;
  T* tmp;
  if (is_device_ptr) {
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
    cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost);
  } else {
    tmp = ptr;
  }

  int print_cols = (k < 0) ? -k : k;
  int start_col = (k < 0) ? stride + k : 0;

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      ss << std::setw(2) << ii << " ";
    } else {
      ss << "   ";
    }

    for (int jj = 0; jj < print_cols; jj += 1) {
      if (ii >= 0) {
        ss << std::setw(7) << std::fixed << std::setprecision(5) 
           << static_cast<float>(tmp[ii * stride + (start_col + jj)]) << " ";
      } else {
        ss << std::setw(7) << (start_col + jj) << " ";
      }
    }
    ss << "\n";
  }
  if (is_device_ptr) {
    free(tmp);
  }
  return ss.str();
}

template std::string getMatrixStr(float* ptr, int m, int k, int stride, bool is_device_ptr);
template std::string getMatrixStr(half* ptr, int m, int k, int stride, bool is_device_ptr);
#ifdef ENABLE_BF16
template std::string getMatrixStr(__nv_bfloat16* ptr, int m, int k, int stride, bool is_device_ptr);
#endif
#ifdef ENABLE_FP8
template std::string getMatrixStr(__nv_fp8_e4m3* ptr, int m, int k, int stride, bool is_device_ptr);
#endif

std::string getMatrixStr(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr) {
  std::stringstream ss;
  unsigned long long* tmp;
  if (is_device_ptr) {
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<unsigned long long*>(malloc(m * stride * sizeof(unsigned long long)));
    cudaMemcpy(tmp, ptr, sizeof(unsigned long long) * m * stride, cudaMemcpyDeviceToHost);
  } else {
    tmp = ptr;
  }

  int print_cols = (k < 0) ? -k : k;
  int start_col = (k < 0) ? stride + k : 0;

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      ss << std::setw(2) << ii << " ";
    } else {
      ss << "   ";
    }

    for (int jj = 0; jj < print_cols; jj += 1) {
      if (ii >= 0) {
        ss << std::setw(4) << tmp[ii * stride + (start_col + jj)] << " ";
      } else {
        ss << std::setw(4) << (start_col + jj) << " ";
      }
    }
    ss << "\n";
  }
  if (is_device_ptr) {
    free(tmp);
  }
  return ss.str();
}

std::string getMatrixStr(int* ptr, int m, int k, int stride, bool is_device_ptr) {
  std::stringstream ss;
  int* tmp;
  if (is_device_ptr) {
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<int*>(malloc(m * stride * sizeof(int)));
    cudaMemcpy(tmp, ptr, sizeof(int) * m * stride, cudaMemcpyDeviceToHost);
  } else {
    tmp = ptr;
  }

  int print_cols = (k < 0) ? -k : k;
  int start_col = (k < 0) ? stride + k : 0;

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      ss << std::setw(2) << ii << " ";
    } else {
      ss << "   ";
    }

    for (int jj = 0; jj < print_cols; jj += 1) {
      if (ii >= 0) {
        ss << std::setw(4) << tmp[ii * stride + (start_col + jj)] << " ";
      } else {
        ss << std::setw(4) << (start_col + jj) << " ";
      }
    }
    ss << "\n";
  }
  if (is_device_ptr) {
    free(tmp);
  }
  return ss.str();
}

std::string getMatrixStr(int64_t* ptr, int m, int k, int stride, bool is_device_ptr) {
  std::stringstream ss;
  int64_t* tmp;
  if (is_device_ptr) {
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<int64_t*>(malloc(m * stride * sizeof(int64_t)));
    cudaMemcpy(tmp, ptr, sizeof(int64_t) * m * stride, cudaMemcpyDeviceToHost);
  } else {
    tmp = ptr;
  }

  int print_cols = (k < 0) ? -k : k;
  int start_col = (k < 0) ? stride + k : 0;

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      ss << std::setw(2) << ii << " ";
    } else {
      ss << "   ";
    }

    for (int jj = 0; jj < print_cols; jj += 1) {
      if (ii >= 0) {
        ss << std::setw(4) << tmp[ii * stride + (start_col + jj)] << " ";
      } else {
        ss << std::setw(4) << (start_col + jj) << " ";
      }
    }
    ss << "\n";
  }
  if (is_device_ptr) {
    free(tmp);
  }
  return ss.str();
}

#ifndef _MSC_VER
std::string getMatrixStr(size_t* ptr, int m, int k, int stride, bool is_device_ptr) {
  std::stringstream ss;
  size_t* tmp;
  if (is_device_ptr) {
    cudaDeviceSynchronize();
    tmp = reinterpret_cast<size_t*>(malloc(m * stride * sizeof(size_t)));
    cudaMemcpy(tmp, ptr, sizeof(size_t) * m * stride, cudaMemcpyDeviceToHost);
  } else {
    tmp = ptr;
  }

  int print_cols = (k < 0) ? -k : k;
  int start_col = (k < 0) ? stride + k : 0;

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      ss << std::setw(2) << ii << " ";
    } else {
      ss << "   ";
    }

    for (int jj = 0; jj < print_cols; jj += 1) {
      if (ii >= 0) {
        ss << std::setw(4) << tmp[ii * stride + (start_col + jj)] << " ";
      } else {
        ss << std::setw(4) << (start_col + jj) << " ";
      }
    }
    ss << "\n";
  }
  if (is_device_ptr) {
    free(tmp);
  }
  return ss.str();
}
#endif

// ============getMatrixStr Ends============

}
