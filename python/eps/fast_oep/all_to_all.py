# pyright: reportCallIssue=false

import torch

from .ops import _ops


class AllToAll:
    def __init__(
        self,
        embed_dim: int,
        size_per_rank: int,
        max_num_global_tokens: int,
        comm
    ) -> None:
        self._ptr = _ops.all_to_all_create(
            embed_dim,
            size_per_rank,
            max_num_global_tokens,
            comm
        )
        assert self._ptr != 0

    def __del__(self) -> None:
        self.destroy()

    def dispatch(
        self,
        ids: torch.Tensor,
    ) -> None:
        assert self._ptr is not None

        _ops.all_to_all_dispatch(
            self._ptr,
            ids,
        )

    def combine(
        self,
        out: torch.Tensor,
        num_global_tokens: int,
        n_grams: int,
        do_permute: bool,
        embed_table: torch.Tensor,
    ) -> None:
        assert self._ptr is not None

        _ops.all_to_all_combine(
            self._ptr,
            out,
            num_global_tokens,
            n_grams,
            do_permute,
            embed_table,
        )

    def destroy(self) -> None:
        if self._ptr is not None:
            _ops.all_to_all_destroy(self._ptr)
            self._ptr = None
