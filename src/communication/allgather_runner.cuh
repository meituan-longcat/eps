#pragma once

#include <sstream>

#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>

namespace eps {

struct AllGatherKernelParams {
  mscclpp::DeviceHandle<mscclpp::SmChannel>* sm_channels;
  int64_t global_rank;
  int64_t attn_tp_size;
  int64_t elems_to_send;
  int64_t local_elems_offset;
  int64_t num_blocks;
  std::string toString() const {
    std::stringstream ss;
    ss << "global_rank=" << global_rank 
       << ", attn_tp_size=" << attn_tp_size 
       << ", elems_to_recv=" << elems_to_send 
       << ", local_elems_offset=" << local_elems_offset;
    return ss.str();
  }
};

template <typename T>
void AllGather(AllGatherKernelParams p, cudaStream_t stream);

class AllGatherRunner {
 public:
  struct Params {
    int64_t global_rank;
    int64_t attn_tp_size;
    int64_t num_blocks;
  };

  explicit AllGatherRunner(Params p) : p_(p) {}

  template <typename T>
  void run(mscclpp::DeviceHandle<mscclpp::SmChannel>* sm_channels, int64_t rows, int64_t cols, cudaStream_t stream) {
    int64_t elems_to_send = 0;
    int64_t local_elems_offset = 0;

    const int rows_per_rank_quotient = rows / p_.attn_tp_size;
    const int rows_per_rank_remainder = rows % p_.attn_tp_size;
    const int tp_rank = p_.global_rank % p_.attn_tp_size;

    if (rows_per_rank_remainder) {
      if (tp_rank < rows_per_rank_remainder) {
        elems_to_send = (rows_per_rank_quotient + 1) * cols;
        local_elems_offset = elems_to_send * tp_rank;
      } else {
        elems_to_send = rows_per_rank_quotient * cols;
        local_elems_offset = (rows_per_rank_quotient + 1) * cols * rows_per_rank_remainder +
                             (tp_rank - rows_per_rank_remainder) * elems_to_send;
      }
    } else {
      elems_to_send = rows_per_rank_quotient * cols;
      local_elems_offset = elems_to_send * tp_rank;
    }

    AllGatherKernelParams p{
      .sm_channels = sm_channels,
      .global_rank = p_.global_rank,
      .attn_tp_size = p_.attn_tp_size,
      .elems_to_send = elems_to_send,
      .local_elems_offset = local_elems_offset,
      .num_blocks = p_.num_blocks
    };

    AllGather<T>(p, stream);
  }

 private:
  Params p_{};
};

}  // namespace eps
