#pragma once
#include <vector>
#include <iostream>
#include <fstream>

#include <cuda_fp16.h>

namespace eps {

void inline validate_config_path(const std::string& path) {
    std::ifstream file(path);
    if(!file) {
        throw std::runtime_error("invalid config_path: " + path);
    }
}

std::string get_hostname();

void copy_print(std::string name, const int *d, int n);

void copy_print(std::string name, const half *d, int n);

void copy_print(std::string name, const float *d, int n);

void syncAndCheckEps(const char *const file, int const line, bool force = false);

template <typename... Args>
std::string fmtstr(const std::string& format, Args... args);

#define sync_check_cuda_error_eps() syncAndCheckEps(__FILE__, __LINE__)
#define force_sync_check_cuda_error_eps() syncAndCheckEps(__FILE__, __LINE__, true)

template <typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr);

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int64_t* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr);

template <typename T>
void print_to_file(
    const T* result,
    const int size,
    const char* file,
    cudaStream_t stream = 0,
    std::ios::openmode open_mode = std::ios::out);

template <typename T>
std::string getMatrixStr(T* ptr, int m, int k, int stride, bool is_device_ptr);

std::string getMatrixStr(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr);
std::string getMatrixStr(int* ptr, int m, int k, int stride, bool is_device_ptr);
std::string getMatrixStr(bool* ptr, int m, int k, int stride, bool is_device_ptr);
std::string getMatrixStr(int64_t* ptr, int m, int k, int stride, bool is_device_ptr);
std::string getMatrixStr(size_t* ptr, int m, int k, int stride, bool is_device_ptr);

}