"""Path Aggregation Network (PANet, Liu et al. 2018) neck'i — DETECTION için bottom-up yol.

FPN/SFP top-down semantik feature yayar; PAN bunun ÜSTÜNE bottom-up bir yol ekler: en ince
seviyeden başlayıp kaba seviyelere doğru stride-2 conv'larla LOKALIZASYON feature'larını yukarı
taşır (YOLOP/YOLOv4 deseni). Girdi ve çıktı aynı 5-seviye sözleşmesi ({"0".."3","pool"}, out_channels
kanal) olduğu için RetinaNet head + anchor'lar değişmeden çalışır.

Yalnız DETECTION yoluna takılır (multitask_model._run_heads); seg/cls base neck'i okumaya devam eder
— tıpkı ASPP'nin yalnız seg'e takılması gibi. Backbone-agnostik: FPN (ResNet) veya SFP (ViT) fark etmez.
Girdi dict'i MUTASYONA UĞRATMAZ (yeni OrderedDict döner) — paylaşılan modda seg/cls aynı base dict'i
görmeye devam etsin diye.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Sequence

import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_LEVELS = ("0", "1", "2", "3", "pool")


def _conv_gn_relu(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.ReLU(inplace=True),
    )


class PANNeck(nn.Module):
    """Bottom-up path aggregation over a 5-level pyramid. `.out_channels` = channels.

    Her seviye geçişinde: N_{i+1} = fuse( downsample_s2(N_i) + P_{i+1} ). En ince seviye (N_0 = P_0)
    değişmeden başlar; lokalizasyon bilgisi kabaya doğru akar. Seviyeler stride'da 2× adımlarla arttığı
    için stride-2 conv doğal olarak eşleşir (güvenlik için boyut uyuşmazlığında interpolate).
    """

    def __init__(self, channels: int = 256, levels: Sequence[str] = DEFAULT_LEVELS):
        super().__init__()
        self.out_channels = channels
        self.levels = tuple(levels)
        n_trans = len(self.levels) - 1
        self.down = nn.ModuleList([_conv_gn_relu(channels, channels, stride=2) for _ in range(n_trans)])
        self.fuse = nn.ModuleList([_conv_gn_relu(channels, channels, stride=1) for _ in range(n_trans)])

    def forward(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = OrderedDict()
        prev = feats[self.levels[0]]
        out[self.levels[0]] = prev  # en ince seviye değişmez
        for i, lvl in enumerate(self.levels[1:]):
            target = feats[lvl]
            d = self.down[i](prev)
            if d.shape[-2:] != target.shape[-2:]:  # tek/çift boyut güvenliği
                d = F.interpolate(d, size=target.shape[-2:], mode="nearest")
            n = self.fuse[i](d + target)
            out[lvl] = n
            prev = n
        return out
