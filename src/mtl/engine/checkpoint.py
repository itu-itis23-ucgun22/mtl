from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
        path,
    )


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str, map_location: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]
