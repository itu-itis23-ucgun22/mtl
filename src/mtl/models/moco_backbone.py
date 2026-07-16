"""MoCo v3 (Momentum Contrast v3, Chen et al. 2021) gövdesi + Simple Feature Pyramid neck'i.
Foundation-model sweep'in "contrastive SSL" temsilcisi (ROADMAP Faz 1).

MoCo v3 = ViT-B/16'yı **contrastive** (pozitif/negatif çift, InfoNCE) hedefle self-supervised eğitir.
Bu, SSL alt-taksonomisinde bize EKSİK olan kutuyu doldurur:
  - self-distillation (DINO)  · **contrastive (MoCo v3)** · masked-pixel (MAE) · masked-latent (I-JEPA)
Böylece "SSL nasıl eğitildi" ekseni dört adil ViT-B/16 temsilcisiyle tamamlanır.

⭐ TAM ADİL: standart timm `vit_base_patch16_224` mimarisi (absolute pos-embed + dynamic_img_size ->
512'de 32x32 grid), ImageNet norm (renorm YOK), sadece AĞIRLIKLAR MoCo v3 (nyu-visionx/moco-v3-vit-b).
DINOv1/DeiT/MAE/CLIP/SAM ile her eksende aynı; tek fark pretraining hedefi (contrastive).

Ağırlık: HF `nyu-visionx/moco-v3-vit-b`. Colab HF indirmesi koparsa MTL_WEIGHTS_DIR/moco_v3_vit_b.safetensors
yerel dosyasından yüklenir (bkz. models/timm_weights.py mantığı; burada inline).
"""
from __future__ import annotations

import os
from typing import Dict

import timm
from torch import Tensor, nn

from mtl.models.sfp import SimpleFeaturePyramid

MOCO_ARCH = "vit_base_patch16_224"      # standart timm ViT-B/16 mimarisi
MOCO_HF = "nyu-visionx/moco-v3-vit-b"   # MoCo v3 SSL ağırlıkları (ImageNet)
MOCO_LOCAL = "moco_v3_vit_b.safetensors"  # MTL_WEIGHTS_DIR içindeki yerel ad (opsiyonel)
PATCH_SIZE = 16
OUT_CHANNELS = 256


def _build_moco_vit(pretrained: bool) -> nn.Module:
    """timm ViT-B/16'yı MoCo v3 ağırlıklarıyla kur (yerel dosya varsa ondan, yoksa HF overlay)."""
    common = dict(num_classes=0, dynamic_img_size=True)
    weights_dir = os.environ.get("MTL_WEIGHTS_DIR")
    local = os.path.join(weights_dir, MOCO_LOCAL) if weights_dir else None

    if pretrained and local and os.path.exists(local):
        from safetensors.torch import load_file
        print(f"[moco] yerel ağırlık kullanılıyor: {local}")
        vit = timm.create_model(MOCO_ARCH, pretrained=False, **common)
        missing, unexpected = vit.load_state_dict(load_file(local), strict=False)
        if unexpected:
            print(f"[moco] beklenmeyen anahtarlar (atlandı): {list(unexpected)[:4]}")
        return vit
    if pretrained:
        # timm HF hub'dan MoCo v3 ağırlıklarını indirir ve vit_base_patch16_224'e yükler
        return timm.create_model(MOCO_ARCH, pretrained=True, pretrained_cfg_overlay=dict(hf_hub_id=MOCO_HF), **common)
    return timm.create_model(MOCO_ARCH, pretrained=False, **common)  # eval: ağırlık checkpoint'ten


class MocoBackbone(nn.Module):
    """MoCo v3 ViT-B/16 (contrastive SSL) + Simple Feature Pyramid, BackboneWithFPN arayüzü.

    forward(images) -> OrderedDict {"0","1","2","3","pool"} (strides 4/8/16/32/64, 256 kanal).
    trainable_blocks: 0 = ViT donuk (kanonik sweep), N = son N blok + final norm.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
    ):
        super().__init__()
        self.vit = _build_moco_vit(pretrained)
        self.num_prefix_tokens = self.vit.num_prefix_tokens  # CLS token(lar)ı atmak için
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
        b, _, h_img, w_img = images.shape
        tokens = self.vit.forward_features(images)             # (B, prefix+N, embed)
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]   # CLS'i at
        h, w = h_img // PATCH_SIZE, w_img // PATCH_SIZE
        return patch_tokens.transpose(1, 2).reshape(b, -1, h, w)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK gövde çıktısı (B, 768, 32, 32) - feature-caching için."""
        return self._tokens_to_grid(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.sfp(x)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.sfp(self._tokens_to_grid(images))
