"""SAM (Segment Anything Model) image-encoder gövdesi + Simple Feature Pyramid neck'i.
Foundation-model sweep'in "segmentation-native" paradigma temsilcisi (ROADMAP Faz 1).

SAM'ın image encoder'ı ViTDet-tarzı bir ViT'tir (pencereli attention + araya serpiştirilmiş
global attention blokları) ve kendi 256-kanal neck'iyle biter. Diğer omurgalarla (DINO/CLIP)
tutarlı olmak için SAM'ın gövde çıktısını (grid) alıp AYNI Simple Feature Pyramid'i uyguluyoruz;
böylece head'ler/pipeline değişmez, kıyas "SAM feature'ları vs diğerleri"ne odaklanır.

⚠️ SANITY CHECK GEREKİR (yerelde Python yok, Colab'da doğrula):
  1. timm `samvit_base_patch16.sa1b`'nin `forward_features` çıktı ŞEKLİ: (B,C,H,W) mı, (B,H,W,C)
     mi, yoksa token (B,N,C) mı? Kod üçünü de otomatik algılar (bkz. _trunk_raw) ama ilk koşuda
     çıkan (embed_dim, h, w) şeklini kontrol et.
  2. SAM native çözünürlük 1024 (pos-embed 64x64). Burada config img_size (512) ile kuruyoruz;
     timm'in pos-embed'i 32x32'ye yeniden boyutlaması gerekiyor. Hata/gürültü olursa img_size'ı
     1024'e çıkar (SAM_IMG) — daha ağır ama native (pencere hizası bozulmaz).
"""
from __future__ import annotations

from typing import Dict

import timm
import torch
from torch import Tensor, nn

from mtl.models.sfp import SimpleFeaturePyramid

SAM_MODEL = "samvit_base_patch16.sa1b"  # SAM ViT-B image encoder (SA-1B pretrained)
SAM_IMG = 512  # config img_size ile EŞLEŞMELİ (SAM native 1024; sorun olursa 1024 yap)
OUT_CHANNELS = 256


class SamBackbone(nn.Module):
    """SAM image encoder + Simple Feature Pyramid, BackboneWithFPN ile aynı arayüz.

    forward(images) -> OrderedDict {"0".."pool"} (5 seviye, out_channels kanal).
    trainable_blocks: 0 = gövde donuk (kanonik sweep), N = son N blok eğitilebilir.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
        img_size: int = SAM_IMG,
    ):
        super().__init__()
        self.vit = timm.create_model(
            SAM_MODEL, pretrained=pretrained, num_classes=0, img_size=img_size
        )
        self._img_size = img_size
        self._set_trainable_blocks(trainable_blocks)

        # Trunk çıktı kanalını (ve şeklini) dummy forward ile otomatik algıla -> SFP'yi ona göre kur.
        with torch.no_grad():
            feat = self._trunk_raw(torch.zeros(1, 3, img_size, img_size))
        embed_dim = feat.shape[1]
        self.out_channels = out_channels
        self.sfp = SimpleFeaturePyramid(embed_dim, out_channels)

    def _set_trainable_blocks(self, trainable_blocks: int) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        if trainable_blocks and trainable_blocks > 0 and hasattr(self.vit, "blocks"):
            blocks = self.vit.blocks
            for blk in blocks[-min(trainable_blocks, len(blocks)):]:
                for p in blk.parameters():
                    p.requires_grad = True

    def _trunk_raw(self, images: Tensor) -> Tensor:
        """SAM gövde çıktısını (B, C, h, w) grid'e normalize eder (şekil ne gelirse gelsin)."""
        feat = self.vit.forward_features(images)
        if feat.dim() == 3:  # token dizisi (B, N, C) -> kareye reshape
            b, n, c = feat.shape
            s = int(round(n ** 0.5))
            feat = feat.transpose(1, 2).reshape(b, c, s, s)
        elif feat.dim() == 4 and feat.shape[1] == feat.shape[2] and feat.shape[1] != feat.shape[3]:
            feat = feat.permute(0, 3, 1, 2).contiguous()  # (B,H,W,C) -> (B,C,H,W)
        return feat  # aksi halde zaten (B,C,H,W)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK gövde çıktısı (feature-caching için; donukken deterministik)."""
        return self._trunk_raw(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.sfp(x)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.sfp(self._trunk_raw(images))
