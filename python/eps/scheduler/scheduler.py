# pyright: reportCallIssue=false

import torch

from .ops import _ops


class ContextWrapper:
    def __init__(self, context):
        self._context = context
        ws_size = _ops.eps_get_workspace_size_context(context)
        self.ws = torch.empty((1, ws_size), dtype=torch.uint8, device="cuda")
        _ops.eps_set_ws_context(context, self.ws)

    def __del__(self) -> None:
        _ops.eps_destroy_context(self._context)


class Scheduler:
    def __init__(
        self,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        input_scales_dim: int,
        max_num_tokens_per_gpu: int,
        comm,
        local_num_experts: int
    ) -> None:
        self._ptr = _ops.eps_create(
            top_k,
            num_experts,
            hidden_size,
            input_scales_dim,
            max_num_tokens_per_gpu,
            comm,
            local_num_experts
        )
        assert self._ptr != 0

        self.local_num_experts = local_num_experts

    def __del__(self) -> None:
        self.destroy()

    def get_context(self, num_tokens, num_global_tokens, num_stages, input_quanted):
        return ContextWrapper(_ops.eps_get_context(self._ptr, num_tokens, num_global_tokens, num_stages, input_quanted))

    def rendezvous_and_plan(
        self, expert_indices: torch.Tensor, expert_scales: torch.Tensor, num_global_tokens, num_stages, input_quanted
    ):
        num_tokens = expert_indices.shape[0]
        self.context = self.get_context(num_tokens, num_global_tokens, num_stages, input_quanted)
        num_will_recv_unique_tokens = _ops.eps_rendezvous(self._ptr, expert_indices, self.context._context)

        exclusive_sum = torch.empty((self.local_num_experts + 1,), dtype=torch.int32, device=expert_indices.device)
        _ops.eps_plan(self._ptr, expert_indices, expert_scales, exclusive_sum, self.context._context)

        return exclusive_sum, num_will_recv_unique_tokens

    def prologue(
        self,
        expert_scales: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        assert self._ptr is not None

        _ops.eps_prologue(
            self._ptr,
            expert_scales,
            hidden_states,
            None,
            None,
            self.context._context
        )

    def dispatch(
        self,
        stage_idx: int
    ) -> None:
        assert self._ptr is not None

        _ops.eps_dispatch(
            self._ptr,
            self.context._context,
            stage_idx
        )

    def gemm_expand(self, gemm_input: torch.Tensor, stage_idx: int):
        _ops.eps_gemm_expand(self._ptr, gemm_input, self.context._context, stage_idx)

    def local_reduce(self, gemm_output: torch.Tensor, stage_idx: int):
        _ops.eps_local_reduce(self._ptr, gemm_output, self.context._context, stage_idx)

    def combine(
        self,
        stage_idx: int
    ) -> None:
        assert self._ptr is not None

        _ops.eps_combine(
            self._ptr,
            self.context._context,
            stage_idx
        )

    def epilogue(
        self,
        output: torch.Tensor
    ) -> None:
        assert self._ptr is not None

        _ops.eps_epilogue(
            self._ptr,
            output,
            self.context._context,
        )

    def destroy(self) -> None:
        if self._ptr is not None:
            _ops.eps_destroy(self._ptr)
            self._ptr = None
