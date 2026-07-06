from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: str, step: int | None = None
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,  # global adım sayısı - oturumlar arası --resume için (None = eski checkpoint)
        },
        path,
    )


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str, map_location: str = "cpu") -> int:
    """Ağırlık+optimizer state'i yükler ve kaydedilen GLOBAL ADIMı döndürür.

    Böylece scripts/train.py --resume kaldığı adımdan devam edebilir. "step" alanı yoksa
    (eski checkpoint) "epoch" değerine düşer - o durumda adım takibi yaklaşık olur.
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step")
    return step if step is not None else ckpt.get("epoch", 0)
