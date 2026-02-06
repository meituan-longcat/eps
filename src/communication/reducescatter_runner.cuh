#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <sstream>

namespace eps {

struct ReduceScatterKernelParams {
  // global world_size
  mscclpp::DeviceHandle<mscclpp::SmChannel>* sm_channels;
  int64_t global_rank;
  int64_t attn_tp_size;
  int64_t elems_to_recv;
  int64_t total_rows;
  int64_t hidden_size;
  int64_t local_elems_offset;
  int64_t num_blocks;
  std::string toString() const {
    std::stringstream ss;
    ss << "global_rank=" << global_rank 
       << ", attn_tp_size=" << attn_tp_size 
       << ", elems_to_recv=" << elems_to_recv 
       << ", total_rows=" << total_rows 
       << ", hidden_size=" << hidden_size 
       << ", local_elems_offset=" << local_elems_offset;
    return ss.str();
  }
};

template <typename T>
void ReduceScatter(ReduceScatterKernelParams p, cudaStream_t stream);

class ReduceScatterRunner {
 public:
  struct Params {
    int global_rank;
    int attn_tp_size;
    int num_blocks;
  };

  explicit ReduceScatterRunner(Params p) : p_(p) {}

  template <typename T>
  void run(mscclpp::DeviceHandle<mscclpp::SmChannel>* sm_channels, int rows, int cols, cudaStream_t stream) {
    int elems_to_recv = 0;
    int local_elems_offset = 0;

    const int rows_per_rank_quotient = rows / p_.attn_tp_size;
    const int rows_per_rank_remainder = rows % p_.attn_tp_size;
    const int tp_rank = p_.global_rank % p_.attn_tp_size;

    if (rows_per_rank_remainder) {
      if (tp_rank < rows_per_rank_remainder) {
        elems_to_recv = (rows_per_rank_quotient + 1) * cols;
        local_elems_offset = elems_to_recv * tp_rank;
      } else {
        elems_to_recv = rows_per_rank_quotient * cols;
        local_elems_offset = (rows_per_rank_quotient + 1) * cols * rows_per_rank_remainder +
                             (tp_rank - rows_per_rank_remainder) * elems_to_recv;
      }
    } else {
      elems_to_recv = rows_per_rank_quotient * cols;
      local_elems_offset = elems_to_recv * tp_rank;
    }

    ReduceScatterKernelParams p{
        .sm_channels = sm_channels,
        .global_rank = p_.global_rank,
        .attn_tp_size = p_.attn_tp_size,
        .elems_to_recv = elems_to_recv,
        .total_rows = rows,
        .hidden_size = cols,
        .local_elems_offset = local_elems_offset,
        .num_blocks = p_.num_blocks
    };

    ReduceScatter<T>(p, stream);
  }

 private:
  Params p_{};
};

}  // namespace eps
