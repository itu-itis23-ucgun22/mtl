"""Semantic segmentation head: en ince FPN seviyesinden beslenen decoder.

Tap point: FPN level "0" (P2, stride 4) - paylaşılan backbone'un en yüksek çözünürlüklü
feature haritası, maske sınır kalitesi için en iyisi.

İki decoder seçeneği (config: model.seg_neck):
  - "fcn"  : düz FCNHead (varsayılan; tüm mevcut sonuçlar bununla).
  - "aspp" : Atrous Spatial Pyramid Pooling (DeepLabv3) - GÖREV-ÖZEL seg neck (Faz 3 ablasyonu).
             Tek feature haritasından çok-ölçekli BAĞLAM çıkarır (paralel dilated conv'lar); hiyerarşi
             GEREKMEZ -> ViT tek-ölçekli feature'a doğal uyar (PANet/FPN'in aksine).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.segmentation.fcn import FCNHead

FPN_TAP_LEVEL = "0"  # P2, stride 4


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabv3, Chen et al. 2017).

    Tek (B, in_ch, H, W) haritasından çok-ölçekli bağlam:
      - paralel dallar: 1x1 conv + 3x3 atrous conv'lar (dilation 6/12/18 -> farklı alıcı-alan)
      - global-pool dalı: tüm-görüntü bağlamı (AdaptiveAvgPool -> 1x1 -> upsample)
      - hepsini concat -> 1x1 project -> out_ch (+ dropout)
    Küçük batch (4) için GroupNorm (BatchNorm kararsız olurdu). Hiyerarşi istemez -> ViT-uyumlu.
    """

    def __init__(self, in_ch: int, out_ch: int = 256, rates=(6, 12, 18)):
        super().__init__()

        def conv_gn_relu(k: int, dilation: int) -> nn.Sequential:
            pad = dilation if k > 1 else 0
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=False),
                nn.GroupNorm(32, out_ch),
                nn.ReLU(inplace=True),
            )

        # 1x1 dalı + her rate için 3x3 atrous dalı
        self.branches = nn.ModuleList([conv_gn_relu(1, 1)] + [conv_gn_relu(3, r) for r in rates])
        # global bağlam dalı
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )
        # concat -> project
        n_branches = len(rates) + 2  # 1x1 + len(rates) atrous + global-pool
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        gp = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


class SemanticSegHead(nn.Module):
    """FPN level "0" -> decoder -> tam çözünürlüğe upsample.

    seg_neck: "fcn" (düz FCN, varsayılan) | "aspp" (görev-özel çok-ölçekli bağlam).
    """

    def __init__(self, in_channels: int, num_classes: int, neck: str = "fcn"):
        super().__init__()
        if neck == "aspp":
            self.decoder = nn.Sequential(
                ASPP(in_channels, out_ch=256),
                nn.Conv2d(256, num_classes, kernel_size=1),
            )
        elif neck == "fcn":
            self.decoder = FCNHead(in_channels, num_classes)
        else:
            raise NotImplementedError(f"seg_neck '{neck}' bilinmiyor. Desteklenen: 'fcn', 'aspp'.")

    def forward(self, features: Dict[str, Tensor], output_size: tuple) -> Tensor:
        x = self.decoder(features[FPN_TAP_LEVEL])
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
