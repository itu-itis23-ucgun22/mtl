"""Collate function for CocoMultiTaskDataset.

Every image is already resized to the same fixed (img_size, img_size) shape
by datasets/transforms.py, so images can be stacked into a single batch
tensor directly - no padding/ImageList batching logic needed here (that's
normally RetinaNet's `.transform` job, which MultiTaskModel bypasses; see
models/detection_head.py).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor


def collate_fn(batch: List[Tuple[Tensor, Dict]]) -> Tuple[Tensor, List[Dict]]:
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
