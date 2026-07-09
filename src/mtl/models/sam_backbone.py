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
import torch.nn.functional as F
from torch import Tensor, nn

from mtl.models.sfp import SimpleFeaturePyramid

SAM_MODEL = "samvit_base_patch16.sa1b"  # SAM ViT-B image encoder (SA-1B pretrained)
SAM_IMG = 512  # config img_size ile EŞLEŞMELİ; DINOv1/CLIP ile aynı 32x32 grid için 512
OUT_CHANNELS = 256


def _interp_pos_embed(pre: Tensor, cur_shape) -> Tensor:
    """SAM pos_embed'i (1, H, W, C) -> (1, h, w, C) bicubic interpole eder (native 1024 -> 512)."""
    _, h, w, _ = cur_shape
    x = pre.permute(0, 3, 1, 2)  # (1, C, H, W)
    x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()  # (1, h, w, C)


def _interp_rel_pos(pre: Tensor, cur_len: int) -> Tensor:
    """rel_pos tablosunu (L, dim) -> (cur_len, dim) linear interpole eder (global-attention blokları)."""
    x = pre.permute(1, 0).unsqueeze(0)  # (1, dim, L)
    x = F.interpolate(x, size=cur_len, mode="linear", align_corners=False)
    return x.squeeze(0).permute(1, 0).contiguous()  # (cur_len, dim)


class SamBackbone(nn.Module):
    """SAM image encoder + Simple Feature Pyramid, BackboneWithFPN ile aynı arayüz.

    forward(images) -> OrderedDict {"0".."pool"} (5 seviye, out_channels kanal).
    trainable_blocks: 0 = gövde donuk (kanonik sweep), N = son N blok eğitilebilir.

    Pretrained yükleme: SAM ağırlıkları native 1024 (64x64 grid; global-attention bloklarının
    pos_embed 64x64, rel_pos tabloları 127) içindir. 512'de (32x32, rel_pos 63) strict load şekil
    uyuşmazlığı verir. Bu yüzden pretrained pozisyonel parametreleri INTERPOLE ederek yüklüyoruz
    (ViT pos-embed interpolasyonu standart; SAM rel_pos'u runtime'da da interpole eder). Böylece SAM
    diğer ViT'lerle aynı 512/32x32 grid'de kalır -> adil kıyas + küçük cache.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
        img_size: int = SAM_IMG,
    ):
        super().__init__()
        # Modeli hedef çözünürlükte (512) rastgele-init kur; pretrained'i interpole ederek yükle.
        self.vit = timm.create_model(
            SAM_MODEL, pretrained=False, num_classes=0, img_size=img_size
        )
        if pretrained:
            self._load_pretrained_interpolated()
        self._img_size = img_size
        self._set_trainable_blocks(trainable_blocks)

        # Trunk çıktı kanalını (ve şeklini) dummy forward ile otomatik algıla -> SFP'yi ona göre kur.
        with torch.no_grad():
            feat = self._trunk_raw(torch.zeros(1, 3, img_size, img_size))
        embed_dim = feat.shape[1]
        self.out_channels = out_channels
        self.sfp = SimpleFeaturePyramid(embed_dim, out_channels)

    def _load_pretrained_interpolated(self) -> None:
        """Native-1024 pretrained ağırlıkları al, pos_embed + rel_pos'u 512 grid'ine interpole et, yükle."""
        pre = timm.create_model(SAM_MODEL, pretrained=True, num_classes=0).state_dict()
        cur = self.vit.state_dict()
        new = {}
        for k, v in pre.items():
            if k not in cur:
                continue
            if v.shape != cur[k].shape:
                if k == "pos_embed":
                    v = _interp_pos_embed(v, cur[k].shape)
                elif k.endswith("rel_pos_h") or k.endswith("rel_pos_w"):
                    v = _interp_rel_pos(v, cur[k].shape[0])
                else:
                    continue  # beklenmeyen uyuşmazlık -> atla (rastgele init kalır)
            new[k] = v
        missing, unexpected = self.vit.load_state_dict(new, strict=False)
        # pos_embed / rel_pos dışında eksik kalan olmamalı (windowed bloklar zaten eşleşir).
        leftover = [m for m in missing if not (m == "pos_embed" or "rel_pos" in m)]
        if leftover:
            print(f"[SamBackbone] uyarı: pretrained'de bulunamayan parametreler: {leftover[:6]}")

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
