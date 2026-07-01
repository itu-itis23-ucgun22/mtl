"""Multi-label whole-image classification head.

Tap point: FPN level "3" (P5, stride 32, 256 channels) - reuses the exact
same feature dict already computed for detection/segmentation, so this head
adds no extra backbone branch and the model has one truly shared computation
graph. The alternative (tapping the raw backbone.body / ResNet layer4
output, pre-FPN, 2048 channels) fully decouples classification gradients
from the FPN's lateral/top-down pathway, at the cost of an extra backbone
branch; see plan doc "Mimari detaylar" for the full tradeoff writeup. Kept
as a documented, cheap-to-add alternative (`cls_head_tap="backbone_body"`)
rather than built now.
"""
from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

FPN_TAP_LEVEL = "3"  # P5, stride 32


class MultiLabelClsHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        x = features[FPN_TAP_LEVEL]
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))
