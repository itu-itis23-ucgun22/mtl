"""Joint multi-task loss: a fixed weighted sum of the four per-task losses.

Adaptive/uncertainty weighting (Kendall, Gal & Cipolla, "Multi-Task Learning
Using Uncertainty to Weigh Losses", CVPR 2018 - learn per-task log-variance
parameters and weight each loss as L_i / (2*sigma_i^2) + log(sigma_i)) is a
natural v2 improvement, deferred here to keep v1 easy to debug.
"""
from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor

from mtl.config import LossConfig


def combine_losses(loss_dict: Dict[str, Tensor], weights: LossConfig) -> Tuple[Tensor, Dict[str, float]]:
    total = (
        weights.det_cls * loss_dict["classification"]
        + weights.det_box * loss_dict["bbox_regression"]
        + weights.seg * loss_dict["seg_loss"]
        + weights.cls * loss_dict["cls_loss"]
    )
    raw = {k: v.item() for k, v in loss_dict.items()}
    raw["total"] = total.item()
    return total, raw
