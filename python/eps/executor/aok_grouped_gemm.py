from typing import Optional

import torch

from .ops import _ops

SM_COUNT: Optional[int] = None


class AokGroupedGemm:
    def __init__(
        self,
        num_local_experts: int
    ) -> None:
        global SM_COUNT
        if SM_COUNT is None:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            SM_COUNT = props.multi_processor_count

        self._ptr = _ops.aok_create(num_local_experts, SM_COUNT)

        assert self._ptr != 0

    def __del__(self) -> None:
        self.destroy()

    def get_workspace_size(self, cols, max_num_tokens):
        return _ops.aok_get_workspace_size(self._ptr, cols, max_num_tokens)

    def run(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        exclusive_sum: torch.Tensor,
        num_tokens_hint: int
    ) -> torch.Tensor:
        assert self._ptr is not None
        assert B_scale is None

        max_num_tokens, K = A.shape
        ws_size = self.get_workspace_size(K, max_num_tokens) * 2
        ws = torch.empty((1, ws_size), dtype=torch.uint8, device=A.device)

        C = torch.empty((max_num_tokens, B.shape[1]), dtype=A.dtype, device=A.device)
        _ops.aok_run(
            self._ptr,
            A,
            B,
            C,
            exclusive_sum,
            num_tokens_hint,
            max_num_tokens,
            ws
        )

        return C

    def destroy(self) -> None:
        if self._ptr is not None:
            _ops.aok_destroy(self._ptr)
            self._ptr = None
