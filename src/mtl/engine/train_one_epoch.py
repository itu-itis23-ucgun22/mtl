from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mtl.config import LossConfig
from mtl.engine.checkpoint import save_checkpoint
from mtl.losses.joint_loss import combine_losses
from mtl.utils.logging import CsvLogger


def _move_targets(targets, device): # tensor listesini GPU ya taşır
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
    checkpoint_every_steps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> int:
    """Runs until the dataloader is exhausted or max_steps is hit.

    If checkpoint_every_steps is set, saves model+optimizer state every that
    many steps (not just at epoch end) - a full epoch can be thousands of
    steps on a real Colab run, and losing all of that to a disconnect or a
    manual interrupt is expensive. The step-tagged checkpoint (`{run_name}
    _step{N}.pt`) can be resumed with `scripts/train.py --resume`.
    """
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    step = start_step

    for images, targets in dataloader:
        images = images.to(device)
        targets = _move_targets(targets, device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"): # 16 bite çevirip geri iş bittikten sonra eski haline çevirmek için kullanılır
            loss_dict = model(images, targets) # model tahmin yapmaya başladı 
            total_loss, raw = combine_losses(loss_dict, loss_cfg) # burada unceratainty i kullanabiliriz gelecekte

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {raw}")

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update() # ölçekleme katsayısını arttırıyor azaltıyor. Eğer loss çok küçükse katsayıyı arttırıyor, çok büyükse azaltıyor. Bu sayede 16 bitte kaybolan hassasiyetin önüne geçiyor.

        if step % log_every == 0:
            logger.log(step, raw)

        step += 1

        if checkpoint_every_steps and step % checkpoint_every_steps == 0:
            path = Path(checkpoint_dir) / f"{run_name}_step{step}.pt"
            save_checkpoint(model, optimizer, step, str(path), step=step)
            print(f"[checkpoint] saved {path}")

        if max_steps is not None and (step - start_step) >= max_steps:
            break

    return step
