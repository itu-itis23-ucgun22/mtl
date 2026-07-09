"""I-JEPA (Image Joint-Embedding Predictive Architecture, Assran et al. 2023) gövdesi
+ Simple Feature Pyramid neck'i. Foundation-model sweep'in "predictive SSL" temsilcisi
(ROADMAP Faz 1, opsiyonel). I-JEPA maskeli-latent tahminiyle eğitilir (kontrastif/distillation
değil) - farklı bir SSL paradigması.

⚠️ İKİ ÖNEMLİ UYARI (diğer omurgalardan farklı):
  1. **timm'de temiz tag yok -> HuggingFace `transformers` gerekir** (`IJepaModel`). Notebook
     hücresi `pip install transformers` yapar. Ağırlık HF'ten iner (`facebook/ijepa_vith14_1k`).
  2. **Sadece ViT-H (632M) var** - diğer omurgalar ViT-B (~86M). Yani bu bir BOYUT CONFOUND'u:
     I-JEPA daha iyi çıkarsa "paradigma mı yoksa 7x daha büyük model mi" ayrışmaz. Sonucu bu notla
     raporla. Ayrıca ViT-H feature-cache'i ~80 GB (patch14@518, 1369 token x1280) -> /content'e
     sığmayabilir; sığmazsa cache yerine düz scripts/train.py (yavaş ama disk dostu) ya da küçük subset.

Normalizasyon: I-JEPA ImageNet norm'uyla eğitildi -> veri hattının ImageNet norm'u uygun,
CLIP'teki gibi yeniden ölçekleme GEREKMEZ. patch14 -> img_size 14'e bölünebilmeli (DINOv2 gibi 518).
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from mtl.models.sfp import SimpleFeaturePyramid

IJEPA_MODEL = "facebook/ijepa_vith14_1k"  # I-JEPA ViT-H/14 (ImageNet-1k)
IJEPA_PATCH = 14
IJEPA_IMG = 518  # 14'e bölünebilir (37x37 grid), DINOv2 ile aynı çözünürlük
OUT_CHANNELS = 256


class IjepaBackbone(nn.Module):
    """I-JEPA ViT-H/14 (HF) + Simple Feature Pyramid, BackboneWithFPN ile aynı arayüz.

    trainable_blocks: 0 = gövde donuk (kanonik sweep - ÖNERİLEN). >0 çözük eğitim HF katman
    yapısına özgü olduğu için burada desteklenmez (donuk sweep'te gerekmiyor).
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
        img_size: int = IJEPA_IMG,
    ):
        super().__init__()
        try:
            from transformers import IJepaModel  # lazy: transformers sadece I-JEPA için gereksin
        except ImportError as e:
            raise ImportError(
                "I-JEPA için `transformers` gerekli: pip install transformers"
            ) from e

        if not pretrained:
            from transformers import IJepaConfig
            self.vit = IJepaModel(IJepaConfig())  # rastgele init (test/offline)
        else:
            self.vit = IJepaModel.from_pretrained(IJEPA_MODEL)

        self._img_size = img_size
        for p in self.vit.parameters():
            p.requires_grad = False
        if trainable_blocks and trainable_blocks > 0:
            raise NotImplementedError(
                "I-JEPA çözük eğitim (trainable_blocks>0) bu wrapper'da yok; donuk sweep (layers=0) kullan."
            )

        # Trunk kanalını dummy forward ile algıla (ViT-H -> 1280) -> SFP'yi ona göre kur.
        with torch.no_grad():
            feat = self._trunk_raw(torch.zeros(1, 3, img_size, img_size))
        embed_dim = feat.shape[1]
        self.out_channels = out_channels
        self.sfp = SimpleFeaturePyramid(embed_dim, out_channels)

    def _trunk_raw(self, images: Tensor) -> Tensor:
        """I-JEPA patch token'larını (B, C, h, w) grid'e çevirir (I-JEPA CLS'siz)."""
        out = self.vit(pixel_values=images, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state  # (B, N, C)
        b, n, c = tokens.shape
        h = w = self._img_size // IJEPA_PATCH
        if n != h * w:  # olası prefix token(lar) olursa sondan h*w patch'i al
            tokens = tokens[:, n - h * w:, :]
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK gövde çıktısı (feature-caching için; donukken deterministik)."""
        return self._trunk_raw(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.sfp(x)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.sfp(self._trunk_raw(images))
