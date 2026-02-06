import torch

from .ops import _ops

def silu(gate_up: torch.Tensor, exclusive_sum: torch.Tensor, num_tokens_hint):
    assert gate_up.dtype == torch.bfloat16

    result = torch.empty((gate_up.shape[0], gate_up.shape[1] // 2), dtype=gate_up.dtype, device=gate_up.device)

    _ops.silu_run(gate_up, result, exclusive_sum, num_tokens_hint)
    return result
