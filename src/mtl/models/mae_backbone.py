"""MAE (Masked Autoencoder, He et al. 2022) gövdesi + Simple Feature Pyramid neck'i.
Foundation-model sweep'in "maskeli yeniden-kurma (MIM)" paradigma temsilcisi (ROADMAP Faz 1).

MAE, görüntünün %75'ini maskeleyip maskeli patch'lerin PİKSELLERİNİ yeniden kurmayı öğrenir.
Bu, sweep'teki diğer SSL ailelerinden ayrı bir paradigma:
  - DINO   : self-distillation (öğrenci-öğretmen, augmentation invariance)
  - CLIP   : image-text contrastive (dil süpervizyonu)
  - SAM    : maske tahmini (segmentation-native, etiketli)
  - MAE    : maskeli piksel yeniden-kurma (generative/reconstructive SSL)   <- BU
  - I-JEPA : maskeli LATENT tahmin (yalnızca ViT-H var -> boyut confound'u, bkz. ijepa_backbone.py)

⭐ NEDEN ADİL: ViT-B/16 @ 512 -> 32x32 grid. Yani DINOv1, CLIP ve SAM ile
  - aynı mimari boyutu (ViT-B, ~86M),
  - aynı patch (16),
  - aynı grid (32x32),
  - aynı normalizasyon (ImageNet; CLIP'teki gibi renorm GEREKMEZ),
  - aynı neck (SFP), aynı head'ler, aynı batch (4).
Tek değişken: PRETRAINING. Sweep'in en temiz eklemesi. (I-JEPA'nın ViT-B'si yayınlanmadığı için
"maskeli tahmin" ailesini adil şekilde ancak MAE temsil edebiliyor.)

Not: `vit_base_patch16_224.mae` = SAF MAE pretrain (ImageNet supervised fine-tune YOK). Supervised
fine-tune'lu varyantı kullanmak paradigma ayrımını bozardı.
"""
from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

from mtl.models.sfp import SimpleFeaturePyramid
from mtl.models.timm_weights import create_timm_model

MAE_MODEL = "vit_base_patch16_224.mae"  # saf MAE pretrain (in1k fine-tune DEĞİL)
PATCH_SIZE = 16
OUT_CHANNELS = 256


class MaeBackbone(nn.Module):
    """MAE ViT-B/16 + Simple Feature Pyramid, BackboneWithFPN ile aynı arayüz.

    forward(images) -> OrderedDict {"0","1","2","3","pool"} (strides 4/8/16/32/64, 256 kanal).
    trainable_blocks: 0 = ViT tamamen donuk (kanonik sweep), N = son N blok + final norm.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
    ):
        super().__init__()
        # create_timm_model: MTL_WEIGHTS_DIR set + yerel .safetensors varsa oradan yükler,
        # yoksa normal timm/HF yolu (bkz. models/timm_weights.py - Colab'da HF kopmalarina karsi).
        self.vit = create_timm_model(
            MAE_MODEL,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,  # 224 disi girislerde pos-embed interpolasyonu (512 -> 32x32)
        )
        self.num_prefix_tokens = self.vit.num_prefix_tokens  # CLS token(lar)i atmak icin
        self.out_channels = out_channels
        self._set_trainable_blocks(trainable_blocks)
        self.sfp = SimpleFeaturePyramid(self.vit.embed_dim, out_channels)  # embed_dim = 768

    def _set_trainable_blocks(self, trainable_blocks: int) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        if trainable_blocks and trainable_blocks > 0:
            blocks = self.vit.blocks
            for blk in blocks[-min(trainable_blocks, len(blocks)):]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.vit.norm.parameters():
                p.requires_grad = True

    def _tokens_to_grid(self, images: Tensor) -> Tensor:
        """ViT patch token'larini (B, embed, h, w) uzamsal haritaya cevirir."""
        b, _, h_img, w_img = images.shape
        tokens = self.vit.forward_features(images)             # (B, prefix+N, embed)
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]   # CLS'i at -> (B, N, embed)
        h, w = h_img // PATCH_SIZE, w_img // PATCH_SIZE
        return patch_tokens.transpose(1, 2).reshape(b, -1, h, w)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK ViT govdesinin ciktisi: (B, 768, 32, 32). Feature-caching icin ayrildi
        (donukken deterministik -> bir kez hesaplanip diske yazilabilir)."""
        return self._tokens_to_grid(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        """EGITILEBILIR neck (Simple Feature Pyramid): trunk grid -> 5-seviye piramit."""
        return self.sfp(x)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.sfp(self._tokens_to_grid(images))
