# pyright: reportCallIssue=false

import logging
import os

import torch

try:
    _lib_path = os.path.join(os.path.dirname(__file__), "libfast_ep.so")
    torch.ops.load_library(_lib_path)
    _ops = torch.ops.fast_ep
except OSError as e:
    from types import SimpleNamespace

    _ops = SimpleNamespace()
    logging.exception(f"Error loading fast_ep: {e}")
