"""Paylaşılan Simple Feature Pyramid (ViTDet, Li et al. 2022) neck'i.

dino_backbone.py'deki neck ile birebir aynı; SAM/I-JEPA gibi yeni ViT omurgaların
tekrar tekrar kopyalamaması için ayrı modül. Tek stride-16 grid'den 5 seviyeli piramit
({"0","1","2","3","pool"}, strides 4/8/16/32/64, hepsi out_channels kanal) türetir.
embed_dim dışarıdan verilir (backbone'a göre değişir: ViT-B 768, ViT-H 1280, SAM-neck 256).

Not: DINOv1/DINOv2/CLIP kendi SFP'lerini inline taşımaya devam ediyor (baseline'ları
bozmamak için dokunulmadı); bu modül yalnızca YENİ omurgalar (sam/ijepa) içindir.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn


def out_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """lateral(1x1)+output(3x3) bloğu, GroupNorm'lu (sıfırdan eğitilen neck için)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
    )


class SimpleFeaturePyramid(nn.Module):
    """(B, embed_dim, h, w) tek grid -> OrderedDict 5 seviye. `.out_channels` = out_channels.

    embed_dim 4'e bölünebilmeli (up4 embed//4 kanala iner): 768/1280/256 hepsi uygun.
    """

    def __init__(self, embed_dim: int, out_channels: int = 256):
        super().__init__()
        assert embed_dim % 4 == 0, f"embed_dim {embed_dim} 4'e bölünebilmeli (SFP up4 için)"
        self.out_channels = out_channels
        # "0" stride 4: 4x upsample (iki ConvTranspose2d)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(32, embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
        )
        self.out0 = out_conv(embed_dim // 4, out_channels)
        # "1" stride 8: 2x upsample
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.out1 = out_conv(embed_dim // 2, out_channels)
        # "2" stride 16: identity ölçek
        self.out2 = out_conv(embed_dim, out_channels)
        # "3" stride 32: 2x downsample (maxpool, parametresiz - ViTDet gibi)
        self.out3 = out_conv(embed_dim, out_channels)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        p0 = self.out0(self.up4(x))                               # stride 4
        p1 = self.out1(self.up2(x))                               # stride 8
        p2 = self.out2(x)                                         # stride 16
        p3 = self.out3(F.max_pool2d(x, kernel_size=2, stride=2))  # stride 32
        pool = F.max_pool2d(p3, kernel_size=1, stride=2, padding=0)  # stride 64
        return OrderedDict([("0", p0), ("1", p1), ("2", p2), ("3", p3), ("pool", pool)])
