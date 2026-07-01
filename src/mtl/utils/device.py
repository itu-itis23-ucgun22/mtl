"""Central CPU/CUDA resolution point.

This is what lets the exact same config file structure drive both a local
CPU smoke test and a Colab GPU run: swap `train.device` in the YAML (or via
--overrides), and this function does the fallback + a clear log message if
CUDA was requested but isn't available (e.g. running a Colab config
locally with --smoke).
"""
from __future__ import annotations

import torch


def resolve_device(preferred: str) -> torch.device:
    if preferred == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but not available - falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)
