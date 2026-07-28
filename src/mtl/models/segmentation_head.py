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


class LRASPP(nn.Module):
    """Lite Reduced ASPP (MobileNetV3, Howard et al. 2019) — HAFİF seg neck.

    ASPP'nin pahalı dilated 3x3 conv'larını ATAR; yerine ucuz **global-bağlam attention kapısı** +
    düşük-seviye skip koyar. Yalnız 1x1 conv + pooling -> edge/mobil için tasarlanmış, çok ucuz.
      high (kaba, stride 16): 1x1 conv (cbr) × global-pool-sigmoid (attention gate) -> upsample
      low  (ince, stride 4) : 1x1 skip
      logits = high_classifier(gated) + low_classifier(low)
    Küçük batch için GroupNorm (BatchNorm yerine). Çok-ÖLÇEK yok (sadece global+yerel) -> ASPP'den
    daha az bağlam, çok daha ucuz. "Bağlam vs maliyet" ablasyonunun hafif ucu.
    """

    def __init__(self, low_ch: int, high_ch: int, num_classes: int, inter_ch: int = 128):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_ch, inter_ch, 1, bias=False),
            nn.GroupNorm(32, inter_ch),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(  # global-bağlam attention (kanal geçidi)
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_ch, inter_ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_ch, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_ch, num_classes, 1)

    def forward(self, low: Tensor, high: Tensor) -> Tensor:
        x = self.cbr(high) * self.scale(high)  # global bağlamla geçitle
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.low_classifier(low) + self.high_classifier(x)


class SemanticSegHead(nn.Module):
    """FPN seviyelerinden decoder -> tam çözünürlüğe upsample.

    seg_neck:
      "fcn"    - düz FCN (varsayılan; level "0").
      "aspp"   - çok-ölçekli bağlam, ağır (level "0").
      "lraspp" - hafif global-bağlam + skip (low="0" stride4, high="2" stride16).
    """

    LRASPP_LOW = "0"   # stride 4 (ince detay skip)
    LRASPP_HIGH = "2"  # stride 16 (kaba bağlam; global-pool burada ucuz)

    def __init__(self, in_channels: int, num_classes: int, neck: str = "fcn"):
        super().__init__()
        self.neck = neck
        if neck == "aspp":
            self.decoder = nn.Sequential(
                ASPP(in_channels, out_ch=256),
                nn.Conv2d(256, num_classes, kernel_size=1),
            )
        elif neck == "lraspp":
            self.decoder = LRASPP(in_channels, in_channels, num_classes)
        elif neck == "fcn":
            self.decoder = FCNHead(in_channels, num_classes)
        else:
            raise NotImplementedError(f"seg_neck '{neck}' bilinmiyor. Desteklenen: 'fcn', 'aspp', 'lraspp'.")

    def forward(self, features: Dict[str, Tensor], output_size: tuple) -> Tensor:
        if self.neck == "lraspp":
            x = self.decoder(features[self.LRASPP_LOW], features[self.LRASPP_HIGH])
        else:
            x = self.decoder(features[FPN_TAP_LEVEL])
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
