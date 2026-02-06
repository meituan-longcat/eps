# pyright: reportCallIssue=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from .ops import _ops


def _as_uint8_cpu(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype != torch.uint8:
            t = t.to(dtype=torch.uint8)
        return t.contiguous()
    t = torch.as_tensor(value, dtype=torch.uint8)
    if t.device.type != "cpu":
        t = t.cpu()
    return t.contiguous()


def _stream_ptr(stream) -> int:
    if stream is None:
        return torch.cuda.current_stream().cuda_stream
    if isinstance(stream, torch.cuda.Stream):
        return stream.cuda_stream
    return int(stream)


@dataclass
class MscclppCommunicatorParams:
    rank: int
    ep_world_size: int
    num_ranks_per_node: int


class MscclppCommunicator:
    def __init__(self, unique_id, p: MscclppCommunicatorParams) -> None:
        uid = _as_uint8_cpu(unique_id)
        self._ptr = int(
            _ops.eps_comm_create(uid, int(p.rank), int(p.ep_world_size), int(p.num_ranks_per_node))
        )
        if self._ptr == 0:
            raise RuntimeError("Failed to create MscclppCommunicator")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", 0)
        if ptr:
            try:
                _ops.eps_comm_destroy(ptr)
            except Exception:
                pass
            self._ptr = 0

    @staticmethod
    def createUniqueId():
        return _ops.eps_comm_create_unique_id()

    def barrier(self) -> None:
        _ops.eps_comm_barrier(self._ptr)

    def data_ptr(self) -> int:
        return int(self._ptr)

    def __int__(self) -> int:
        return int(self._ptr)


def sync_check_cuda_error_eps() -> None:
    _ops.sync_check_cuda_error_eps()


class TPDPConvertor:
    @dataclass
    class Params:
        global_rank: int
        tp_max_num_tokens: int
        attn_tp_size: int
        hidden_size: int
        comm: Union[int, "MscclppCommunicator"]

        def comm_ptr(self) -> int:
            if isinstance(self.comm, MscclppCommunicator):
                return self.comm.data_ptr()
            return int(self.comm)

    class ReduceScatterContext:
        def __init__(self, tpdp: "TPDPConvertor", tp_num_tokens: int, num_blocks: int) -> None:
            self._tpdp = tpdp
            self.tp_num_tokens = int(tp_num_tokens)
            self.num_blocks = int(num_blocks)

        def input(self):
            return _ops.eps_tpdp_get_reduce_scatter_input(
                self._tpdp._ptr, self.tp_num_tokens, self.num_blocks
            )

        def output(self):
            return _ops.eps_tpdp_get_reduce_scatter_output(
                self._tpdp._ptr, self.tp_num_tokens, self.num_blocks
            )

        @property
        def output_row_offset(self) -> int:
            return int(
                _ops.eps_tpdp_get_reduce_scatter_output_row_offset(
                    self._tpdp._ptr, self.tp_num_tokens, self.num_blocks
                )
            )

    class AllGatherContext:
        def __init__(
            self, tpdp: "TPDPConvertor", tp_num_tokens: int, hidden_size: int, num_blocks: int
        ) -> None:
            self._tpdp = tpdp
            self.tp_num_tokens = int(tp_num_tokens)
            self.hidden_size = int(hidden_size)
            self.num_blocks = int(num_blocks)

        def input(self):
            return _ops.eps_tpdp_get_all_gather_input(
                self._tpdp._ptr, self.tp_num_tokens, self.hidden_size, self.num_blocks
            )

        def output(self):
            return _ops.eps_tpdp_get_all_gather_output(
                self._tpdp._ptr, self.tp_num_tokens, self.hidden_size, self.num_blocks
            )

        @property
        def input_row_offset(self) -> int:
            return int(
                _ops.eps_tpdp_get_all_gather_input_row_offset(
                    self._tpdp._ptr, self.tp_num_tokens, self.hidden_size, self.num_blocks
                )
            )

    def __init__(self, p: "TPDPConvertor.Params") -> None:
        self._ptr = int(
            _ops.eps_tpdp_create(
                int(p.global_rank),
                int(p.tp_max_num_tokens),
                int(p.attn_tp_size),
                int(p.hidden_size),
                p.comm_ptr(),
            )
        )
        if self._ptr == 0:
            raise RuntimeError("Failed to create TPDPConvertor")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", 0)
        if ptr:
            try:
                _ops.eps_tpdp_destroy(ptr)
            except Exception:
                pass
            self._ptr = 0

    def data_ptr(self) -> int:
        return int(self._ptr)

    def __int__(self) -> int:
        return int(self._ptr)

    def get_reduce_scatter_context(self, tp_num_tokens: int, num_blocks: int = 45):
        return TPDPConvertor.ReduceScatterContext(self, tp_num_tokens, num_blocks)

    def reduce_scatter(self, context: "TPDPConvertor.ReduceScatterContext", stream) -> None:
        _ops.eps_tpdp_reduce_scatter(
            self._ptr, context.tp_num_tokens, context.num_blocks, _stream_ptr(stream)
        )

    def get_all_gather_context(self, tp_num_tokens: int, hidden_size: int, num_blocks: int = 28):
        return TPDPConvertor.AllGatherContext(self, tp_num_tokens, hidden_size, num_blocks)

    def all_gather(self, context: "TPDPConvertor.AllGatherContext", stream) -> None:
        _ops.eps_tpdp_all_gather(
            self._ptr,
            context.tp_num_tokens,
            context.hidden_size,
            context.num_blocks,
            _stream_ptr(stream),
        )
