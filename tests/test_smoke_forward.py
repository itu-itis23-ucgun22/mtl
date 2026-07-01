"""Overfit-one-batch sanity check: catches wiring bugs (detached graph,
wrong target dtype, shape mismatches) far more reliably than "did it crash".
Uses pretrained=False so it needs no network access / weight download.
"""
from __future__ import annotations

import torch

from mtl.config import LossConfig
from mtl.losses.joint_loss import combine_losses
from mtl.models.multitask_model import MultiTaskModel

NUM_CLASSES = 5
IMG_SIZE = 128


def _make_batch(batch_size: int = 2):
    images = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE)
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
            "labels": torch.tensor([0], dtype=torch.int64),
            "sem_mask": torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.int64),
            "cls_labels": torch.zeros(NUM_CLASSES, dtype=torch.float32),
        }
        for _ in range(batch_size)
    ]
    return images, targets


def test_overfit_one_batch():
    torch.manual_seed(0)
    model = MultiTaskModel(
        pretrained=False,
        det_num_classes=NUM_CLASSES,
        seg_num_classes=NUM_CLASSES + 1,
        cls_num_labels=NUM_CLASSES,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_cfg = LossConfig()
    images, targets = _make_batch()

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        total, raw = combine_losses(loss_dict, loss_cfg)
        assert torch.isfinite(total), raw
        total.backward()
        optimizer.step()
        losses.append(total.item())

    assert losses[-1] < losses[0], losses
