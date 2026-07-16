"""BEiT (Bao et al. 2021) gövdesi + Simple Feature Pyramid neck'i. Foundation-model sweep'in
"masked discrete-token" SSL temsilcisi (ROADMAP Faz 1).

BEiT = görüntünün bir kısmını maskeleyip maskeli patch'lerin **discrete görsel token'larını**
(bir dVAE/VQ tokenizer'ın kodlarını) tahmin etmeyi öğrenir. Bu, "maskeli tahminde NE tahmin edilir"
ekseninin üçüncü kardeşidir:
  - piksel yeniden-kur (MAE) · **discrete token tahmin (BEiT)** · latent tahmin (I-JEPA)

⚠️ İKİ UYARI (diğer adil-çekirdek üyelerinden farklı):
  1. **Saf-SSL sürümü HF transformers'ta** (`microsoft/beit-base-patch16-224-pt22k`; timm'de yalnız
     fine-tune'lu tag'ler var). `pip install transformers`. `pt22k` = SSL pretrain (ft YOK) ama
     **ImageNet-22k** üzerinde -> diğerleri IN1k, hafif VERİ confound'u (not düş).
  2. **BEiT relative-position-bias kullanır** -> 512'de (32x32) rel-pos tablosu native 224 (14x14) için;
     SAM'daki gibi interpolasyon gerekir. `interpolate_pos_encoding=True` ile deniyoruz (yeni transformers
     bunu rel-pos'a da uygular). İlk koşuda trunk şekli (768, 32, 32) çıkmazsa/hata verirse -> transformers'ı
     güncelle (`pip install -U transformers`) ya da SAM-tarzı rel-pos interpolasyonu gerekir (sanity-check).

Normalizasyon ImageNet (renorm gerekmez). patch16 -> 512 = 32x32 grid (diğer ViT-B'lerle aynı, adil).
"""
from __future__ import annotations

import os
from typing import Dict

import torch
from torch import Tensor, nn

from mtl.models.sfp import SimpleFeaturePyramid

BEIT_HF = "microsoft/beit-base-patch16-224-pt22k"  # SAF SSL (pretrain-only, ft DEĞİL)
BEIT_LOCAL = "beit-base-patch16-224-pt22k"  # MTL_WEIGHTS_DIR içindeki YEREL KLASÖR adı (opsiyonel)
BEIT_PATCH = 16
BEIT_IMG = 512
OUT_CHANNELS = 256


def _beit_source() -> tuple[str, bool]:
    """(model_kaynağı, yerel_mi) döndürür. MTL_WEIGHTS_DIR/beit-.../ klasörü varsa Colab'ın
    Xet-CDN indirmesine hiç girmeden ORADAN yüklenir; yoksa HF hub adına düşülür."""
    weights_dir = os.environ.get("MTL_WEIGHTS_DIR")
    if weights_dir:
        local = os.path.join(weights_dir, BEIT_LOCAL)
        if os.path.isdir(local):
            print(f"[beit] yerel ağırlık klasörü kullanılıyor: {local}")
            return local, True
    return BEIT_HF, False


class BeitBackbone(nn.Module):
    """BEiT ViT-B/16 (masked discrete-token SSL) + Simple Feature Pyramid.

    trainable_blocks: 0 = donuk (kanonik sweep). >0 çözük eğitim HF katman yapısına özgü olduğu için
    burada desteklenmez (donuk sweep'te gerekmiyor).
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
        img_size: int = BEIT_IMG,
    ):
        super().__init__()
        try:
            from transformers import BeitModel  # lazy: transformers sadece BEiT için gereksin
        except ImportError as e:
            raise ImportError("BEiT için `transformers` gerekli: pip install transformers") from e

        source, is_local = _beit_source()
        # Yerel klasörden yüklerken local_files_only ile HER TÜRLÜ ağ çağrısını (Xet dahil) kapat.
        load_kwargs = dict(add_pooling_layer=False)
        if is_local:
            load_kwargs["local_files_only"] = True

        if not pretrained:
            from transformers import BeitConfig
            # eval/görselleştirme: doğru MİMARİ (ağırlık checkpoint'ten) -> config'i kaynaktan al
            cfg = BeitConfig.from_pretrained(source, local_files_only=is_local)
            self.vit = BeitModel(cfg, add_pooling_layer=False)
        else:
            self.vit = BeitModel.from_pretrained(source, **load_kwargs)

        self._img_size = img_size
        for p in self.vit.parameters():
            p.requires_grad = False
        if trainable_blocks and trainable_blocks > 0:
            raise NotImplementedError(
                "BEiT çözük eğitim (trainable_blocks>0) bu wrapper'da yok; donuk sweep (layers=0) kullan."
            )

        # Trunk kanalını dummy forward ile algıla (ViT-B -> 768) -> SFP'yi ona göre kur.
        with torch.no_grad():
            feat = self._trunk_raw(torch.zeros(1, 3, img_size, img_size))
        embed_dim = feat.shape[1]
        self.out_channels = out_channels
        self.sfp = SimpleFeaturePyramid(embed_dim, out_channels)

    def _trunk_raw(self, images: Tensor) -> Tensor:
        """BEiT patch token'larını (B, C, h, w) grid'e çevirir (CLS token index 0 atılır)."""
        out = self.vit(pixel_values=images, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state  # (B, 1+N, C) - BEiT'te CLS index 0
        # BEiT'in final LayerNorm'u use_mean_pooling=True config'inde pooler'a taşınır; biz pooler'ı
        # (add_pooling_layer=False) kapattığımız için last_hidden_state NORMALİZE EDİLMEMİŞ döner
        # -> devasa aktivasyonlar (std~19, aralık ±500) sıfırdan neck/head öğrenmesini bozar. Diğer
        # ViT'ler kendi final norm'unu içerir; burada parametresiz LayerNorm ile ölçeği düzeltiyoruz.
        tokens = torch.nn.functional.layer_norm(tokens, (tokens.shape[-1],))
        b, n, c = tokens.shape
        h = w = self._img_size // BEIT_PATCH
        # CLS + olası ekstra prefix'i at: sondan h*w patch token'ı al
        patch = tokens[:, n - h * w:, :] if n != h * w else tokens
        return patch.transpose(1, 2).reshape(b, c, h, w)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK gövde çıktısı - feature-caching için (donukken deterministik)."""
        return self._trunk_raw(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.sfp(x)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.sfp(self._trunk_raw(images))
