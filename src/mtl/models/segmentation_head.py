"""Semantic segmentation head: a plain FCN head fed by the finest FPN level.

Tap point: FPN level "0" (P2, stride 4) - the highest-resolution feature map
available from the shared backbone, giving the best mask boundary quality
among the four FPN levels.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.segmentation.fcn import FCNHead

FPN_TAP_LEVEL = "0"  # P2, stride 4


class SemanticSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.fcn = FCNHead(in_channels, num_classes)

    def forward(self, features: Dict[str, Tensor], output_size: tuple) -> Tensor:
        x = features[FPN_TAP_LEVEL]
        x = self.fcn(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
