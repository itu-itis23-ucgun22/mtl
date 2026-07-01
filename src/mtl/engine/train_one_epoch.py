from __future__ import annotations

from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mtl.config import LossConfig
from mtl.losses.joint_loss import combine_losses
from mtl.utils.logging import CsvLogger


def _move_targets(targets, device):
    moved = []
    for t in targets:
        moved.append({k: v.to(device) for k, v in t.items()})
    return moved


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_cfg: LossConfig,
    device: torch.device,
    logger: CsvLogger,
    max_steps: Optional[int] = None,
    log_every: int = 10,
    amp: bool = False,
    start_step: int = 0,
) -> int:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    step = start_step

    for images, targets in dataloader:
        images = images.to(device)
        targets = _move_targets(targets, device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            total_loss, raw = combine_losses(loss_dict, loss_cfg)

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {raw}")

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0:
            logger.log(step, raw)

        step += 1
        if max_steps is not None and (step - start_step) >= max_steps:
            break

    return step
