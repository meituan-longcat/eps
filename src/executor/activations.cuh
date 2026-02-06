#pragma once

#include "src/scheduler/gemm_executor_interface.cuh"

#include "cuda_runtime.h"

namespace eps {


template<typename T>
void silu_activate(const T* gate_up, T* result_, const int* exclusive_sum, int num_experts, int inter_size_x2, int num_tokens_hint, cudaStream_t stream);


template<typename T>
class Activation : public ActivationInterface {
public:

  void run(RunParams p, cudaStream_t stream) override;
};

template<typename T>
class DenseActivation : public DenseActivationInterface {
public:

  void run(RunParams p, cudaStream_t stream) override;
};

template<typename T>
class ActivationUnfused : public ActivationInterface {
public:

  void run(RunParams p, cudaStream_t stream) override;
};

template <typename T>
void activationLauncher(T *gate, T *up, const int *exclusive_sum,
                        size_t num_experts_per_stage, size_t n,
                        cudaStream_t stream);

template <typename T>
void activationLauncher2(T *activation, T *gate_up,
                         const int *exclusive_sum, size_t num_experts_per_stage,
                         size_t n2, cudaStream_t stream);

}
