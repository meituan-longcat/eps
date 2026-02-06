# pyright: reportCallIssue=false

import torch

from .ops import _ops


class AllToAll:
    def __init__(
        self,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        max_num_global_tokens: int,
        comm
    ) -> None:
        self._ptr = _ops.all_to_all_create(
            top_k,
            num_experts,
            hidden_size,
            max_num_global_tokens,
            comm
        )
        assert self._ptr != 0

    def __del__(self) -> None:
        self.destroy()

    def dispatch(
        self,
        out_exclusive_sum: torch.Tensor,
        out_expert_x: torch.Tensor,
        dp_x: torch.Tensor,
        indices: torch.Tensor,
        num_global_tokens: int
    ) -> None:
        assert self._ptr is not None

        _ops.all_to_all_dispatch(
            self._ptr,
            out_exclusive_sum,
            out_expert_x,
            dp_x,
            indices,
            num_global_tokens
        )

    def combine(
        self,
        out_tokens: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        num_global_tokens: int
    ) -> None:
        assert self._ptr is not None

        _ops.all_to_all_combine(
            self._ptr,
            out_tokens,
            weights,
            expert_y,
            num_global_tokens
        )

    def destroy(self) -> None:
        if self._ptr is not None:
            _ops.all_to_all_destroy(self._ptr)
            self._ptr = None
