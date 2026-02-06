# pyright: reportCallIssue=false

import logging
import os

import torch

try:
    _lib_path = os.path.join(os.path.dirname(__file__), "libexecutor_.so")
    torch.ops.load_library(_lib_path)
    _ops = torch.ops.executor
except OSError as e:
    from types import SimpleNamespace

    _ops = SimpleNamespace()
    logging.exception(f"Error loading eps executor: {e}")
    raise e
