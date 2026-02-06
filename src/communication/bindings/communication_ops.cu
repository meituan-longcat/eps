#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cuda_bf16.h>

#include <cstring>
#include <vector>

#include "src/common/debug.cuh"
#include "src/communication/msccl_comm.cuh"
#include "src/communication/tp_dp_convertor.cuh"

namespace {

using fptr_t = int64_t;
using eps::MscclppCommunicator;

at::Tensor comm_create_unique_id() {
  auto id = MscclppCommunicator::createUniqueId();
  auto options = at::TensorOptions().dtype(at::kByte).device(at::kCPU);
  auto out = at::empty({static_cast<int64_t>(sizeof(id))}, options);
  std::memcpy(out.data_ptr(), &id, sizeof(id));
  return out;
}

fptr_t comm_create(
    const at::Tensor &unique_id,
    int64_t rank,
    int64_t ep_world_size,
    int64_t num_ranks_per_node) {
  TORCH_CHECK(unique_id.device().is_cpu(), "unique_id must be a CPU tensor");
  TORCH_CHECK(unique_id.scalar_type() == at::kByte, "unique_id must be uint8");
  TORCH_CHECK(unique_id.is_contiguous(), "unique_id must be contiguous");
  TORCH_CHECK(
      unique_id.numel() == static_cast<int64_t>(sizeof(mscclpp::UniqueId)),
      "unique_id must be ",
      sizeof(mscclpp::UniqueId),
      " bytes");

  mscclpp::UniqueId id;
  std::memcpy(&id, unique_id.data_ptr(), sizeof(id));

  MscclppCommunicator::Params p{
      static_cast<int>(rank),
      static_cast<int>(ep_world_size),
      static_cast<int>(num_ranks_per_node)};
  auto *comm = new MscclppCommunicator(id, p);
  return reinterpret_cast<fptr_t>(comm);
}

void comm_destroy(fptr_t ptr) { delete reinterpret_cast<MscclppCommunicator *>(ptr); }

void comm_barrier(fptr_t ptr) {
  auto *comm = reinterpret_cast<MscclppCommunicator *>(ptr);
  TORCH_CHECK(comm != nullptr, "comm is null");
  comm->barrier();
}

void sync_check_cuda_error_eps_wrap() { eps::syncAndCheckEps(__FILE__, __LINE__); }

using TPDP = eps::TPDPConvertor<__nv_bfloat16>;

TPDP *tpdp_ptr(fptr_t ptr) { return reinterpret_cast<TPDP *>(ptr); }

at::Tensor make_bf16_view(void *data, int64_t rows, int64_t cols, int64_t device_index) {
  c10::cuda::CUDAGuard device_guard(device_index);
  std::vector<int64_t> sizes{rows, cols};
  std::vector<int64_t> strides{cols, 1};
  auto options =
      at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kBFloat16);
  return at::from_blob(data, sizes, strides, [](void *) {}, options);
}

fptr_t tpdp_create(
    int64_t global_rank,
    int64_t tp_max_num_tokens,
    int64_t attn_tp_size,
    int64_t hidden_size,
    fptr_t comm_ptr) {
  auto *comm = reinterpret_cast<MscclppCommunicator *>(comm_ptr);
  TORCH_CHECK(comm != nullptr, "comm is null");
  eps::TPDPConvertorParams p{
      static_cast<int>(global_rank),
      static_cast<int>(tp_max_num_tokens),
      static_cast<int>(attn_tp_size),
      static_cast<int>(hidden_size),
      comm};
  auto *tpdp = new TPDP(p);
  return reinterpret_cast<fptr_t>(tpdp);
}

void tpdp_destroy(fptr_t ptr) { delete tpdp_ptr(ptr); }

at::Tensor tpdp_get_reduce_scatter_input(fptr_t ptr, int64_t tp_num_tokens, int64_t num_blocks) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx =
      tpdp->get_reduce_scatter_context(static_cast<int>(tp_num_tokens), static_cast<int>(num_blocks));
  int device_index = ctx.rank % ctx.num_ranks_per_node;
  return make_bf16_view(ctx.input, ctx.input_rows, ctx.hidden_size, device_index);
}

at::Tensor tpdp_get_reduce_scatter_output(fptr_t ptr, int64_t tp_num_tokens, int64_t num_blocks) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx =
      tpdp->get_reduce_scatter_context(static_cast<int>(tp_num_tokens), static_cast<int>(num_blocks));
  int device_index = ctx.rank % ctx.num_ranks_per_node;
  return make_bf16_view(ctx.output, ctx.output_rows, ctx.hidden_size, device_index);
}

int64_t tpdp_get_reduce_scatter_output_row_offset(
    fptr_t ptr,
    int64_t tp_num_tokens,
    int64_t num_blocks) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx =
      tpdp->get_reduce_scatter_context(static_cast<int>(tp_num_tokens), static_cast<int>(num_blocks));
  return static_cast<int64_t>(ctx.output_row_offset);
}

void tpdp_reduce_scatter(fptr_t ptr, int64_t tp_num_tokens, int64_t num_blocks, int64_t stream) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx =
      tpdp->get_reduce_scatter_context(static_cast<int>(tp_num_tokens), static_cast<int>(num_blocks));
  tpdp->reduce_scatter(ctx, reinterpret_cast<cudaStream_t>(stream));
}

at::Tensor tpdp_get_all_gather_input(
    fptr_t ptr,
    int64_t tp_num_tokens,
    int64_t hidden_size,
    int64_t num_blocks) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx = tpdp->get_all_gather_context(
      static_cast<int>(tp_num_tokens), hidden_size, static_cast<int>(num_blocks));
  int device_index = ctx.rank % ctx.num_ranks_per_node;
  return make_bf16_view(ctx.input, ctx.input_rows, ctx.hidden_size, device_index);
}

at::Tensor tpdp_get_all_gather_output(
    fptr_t ptr,
    int64_t tp_num_tokens,
    int64_t hidden_size,
    int64_t num_blocks) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx = tpdp->get_all_gather_context(
      static_cast<int>(tp_num_tokens), hidden_size, static_cast<int>(num_blocks));
  int device_index = ctx.rank % ctx.num_ranks_per_node;
  return make_bf16_view(ctx.output, ctx.output_rows, ctx.hidden_size, device_index);
}

int64_t tpdp_get_all_gather_input_row_offset(
    fptr_t ptr,
    int64_t tp_num_tokens,
    int64_t hidden_size,
    int64_t num_blocks) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx = tpdp->get_all_gather_context(
      static_cast<int>(tp_num_tokens), hidden_size, static_cast<int>(num_blocks));
  return static_cast<int64_t>(ctx.input_row_offset);
}

void tpdp_all_gather(
    fptr_t ptr,
    int64_t tp_num_tokens,
    int64_t hidden_size,
    int64_t num_blocks,
    int64_t stream) {
  auto *tpdp = tpdp_ptr(ptr);
  TORCH_CHECK(tpdp != nullptr, "tpdp is null");
  auto ctx = tpdp->get_all_gather_context(
      static_cast<int>(tp_num_tokens), hidden_size, static_cast<int>(num_blocks));
  tpdp->all_gather(ctx, reinterpret_cast<cudaStream_t>(stream));
}

} // namespace

namespace eps {
void register_communication_ops(torch::Library &m) {
  m.def("eps_comm_create_unique_id", &comm_create_unique_id);
  m.def("eps_comm_create", &comm_create);
  m.def("eps_comm_destroy", &comm_destroy);
  m.def("eps_comm_barrier", &comm_barrier);
  m.def("sync_check_cuda_error_eps", &sync_check_cuda_error_eps_wrap);

  m.def("eps_tpdp_create", &tpdp_create);
  m.def("eps_tpdp_destroy", &tpdp_destroy);
  m.def("eps_tpdp_get_reduce_scatter_input", &tpdp_get_reduce_scatter_input);
  m.def("eps_tpdp_get_reduce_scatter_output", &tpdp_get_reduce_scatter_output);
  m.def("eps_tpdp_get_reduce_scatter_output_row_offset", &tpdp_get_reduce_scatter_output_row_offset);
  m.def("eps_tpdp_reduce_scatter", &tpdp_reduce_scatter);
  m.def("eps_tpdp_get_all_gather_input", &tpdp_get_all_gather_input);
  m.def("eps_tpdp_get_all_gather_output", &tpdp_get_all_gather_output);
  m.def("eps_tpdp_get_all_gather_input_row_offset", &tpdp_get_all_gather_input_row_offset);
  m.def("eps_tpdp_all_gather", &tpdp_all_gather);
}
} // namespace eps
