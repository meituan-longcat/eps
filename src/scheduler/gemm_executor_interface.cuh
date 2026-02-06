#pragma once

class GemmInterface {
public:
  struct RunParams {
      const void* A;
      const void* A_scales;
      const void* B;
      const void* weight_scales;
      const void* biases;
      void* C;
      const int* total_rows_before_expert;
      int64_t total_rows;
      int64_t gemm_n;
      int64_t gemm_k;
      int num_experts_per_stage;
      int expert_begin;
      char* ws;
      int64_t max_num_tokens;
  };

  virtual void run(RunParams p, cudaStream_t stream) = 0;

  virtual size_t output_type_bytes() = 0;

  virtual size_t getWorkspaceSize(int64_t cols, int64_t max_num_tokens, int64_t local_num_experts) = 0;

  virtual ~GemmInterface() = default;
};

class ActivationInterface {
public:
    struct RunParams {
        void *activation;
        void *gate_up;
        const int *exclusive_sum;
        int num_experts_per_stage;
        int64_t inter_size_x2;

        void *gate;
        void *up;
        int64_t inter_size;

        int64_t num_tokens_hint;
    };

  virtual void run(RunParams p, cudaStream_t stream) = 0;
};

class DenseActivationInterface {
public:
    struct RunParams {
        void *activation;
        void *gate_up;
        int64_t inter_size_x2;

        int64_t num_tokens;
    };

  virtual void run(RunParams p, cudaStream_t stream) = 0;
};

class GemmExecutorInterface {
public:
    struct RunParams
    {
        const void* input;
        const float* input_scales;
        void* output;
        const int* exclusive_sum; /*[local_expert_end - local_expert_begin + 1]*/
        int num_stages;
        int64_t num_tokens_hint;
        int64_t max_num_tokens;
    };

    // Should satisfy the constraint specified in Context::getWorkspaceSize
    virtual size_t getWorkspaceSize(int64_t max_num_tokens) = 0;

    virtual void run(RunParams rp, int stage_idx, void* ws, cudaStream_t stream) = 0;

    virtual ~GemmExecutorInterface() = default;
};

class DenseGemmInterface {
public:
  struct RunParams {
      const void* A;
      const void* A_scales;
      const void* B;
      const void* weight_scales;
      void* C;
      int64_t gemm_n;
      int64_t gemm_k;
      char* ws;
      int64_t num_tokens;
  };

  virtual void run(RunParams p, cudaStream_t stream) = 0;

  virtual size_t output_type_bytes() = 0;

  virtual size_t getWorkspaceSize(int64_t cols, int64_t num_tokens) = 0;

  virtual ~DenseGemmInterface() = default;
};

class SharedExpertsInterface {
public:
    struct RunParams
    {
        const void* input;
        const void* input_scales;
        void* output;
        int num_tokens;
    };

    virtual size_t getWorkspaceSize(int num_tokens) = 0;

    virtual void run(RunParams rp, void* ws, cudaStream_t stream) = 0;

    virtual ~SharedExpertsInterface() = default;
};

class BatchedGemmInterface {
public:
  struct RunParams {
      const void* A;
      const void* A_scales;
      const void* B;
      const void* weight_scales;
      void* C;
      int64_t M;  // per gemm
      int64_t N;  // per gemm
      int64_t K;  // per gemm
      int64_t A_stride;  // A move to next gemm
      int64_t B_stride;  // B move to next gemm
      int64_t C_stride;  // C move to next gemm
      int64_t lda;       // A move to next row
      int64_t ldb;       // B move to next row
      int64_t ldc;       // C move to next row
      int num_gemms;
      char* ws;
  };

  virtual void run(RunParams rp, cudaStream_t stream) = 0;

  virtual size_t output_type_bytes() = 0;

  virtual size_t getWorkspaceSize(RunParams rp) = 0;

  virtual ~BatchedGemmInterface() = default;
};