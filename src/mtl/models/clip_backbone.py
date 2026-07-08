"""CLIP (image-text contrastive ViT-B/16) gövdesi + ViTDet tarzı Simple Feature
Pyramid neck'i - ResNet+FPN ve DINO omurgalarıyla *aynı çıktı sözleşmesini* üretir.

Neden bu yapı (DINOv1 baseline'ı bozmamak için AYRI dosya; head'ler/pipeline değişmez):
  - CLIP görüntü kodlayıcısı *plain* bir ViT'tir (DINOv1 gibi): tek çözünürlük
    (stride 16) üretir, FPN'in birleştireceği doğal bir hiyerarşi yoktur. Bu yüzden
    FPN yerine ViTDet'in (Li et al. 2022) Simple Feature Pyramid'i kullanılıyor -
    dino_backbone.py ile birebir aynı neck. CLIP ViT-B/16 patch16 olduğu için grid
    DINOv1 ile aynı (512px -> 32x32): "aynı mimari, farklı pretraining (dil-contrastive
    vs SSL)" -> DINOv1 ile en TEMİZ kıyas (patch confound'u yok).

Normalizasyon (ÖNEMLİ): CLIP kendi mean/std'siyle eğitildi (ImageNet'ten farklı). Veri
hattı (datasets/transforms.py) tüm backbone'lara ImageNet-norm veriyor. CLIP'e ImageNet
norm vermek onu hafif haksız yere sokar; bu yüzden gövdeye girmeden ÖNCE burada
ImageNet-norm -> CLIP-norm yeniden ölçekleniyor. Böylece transforms.py'ye dokunulmaz
(diğer backbone'lar etkilenmez) ve her backbone kendi native ön-işlemesini alır.
Bu dönüşüm trunk_forward içinde olduğu için cache'lenen feature'lar da doğrudur.

timm neden: dino_backbone.py'deki gerekçenin aynısı (offline pretrained=False, dynamic_img_size).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

CLIP_MODEL = "vit_base_patch16_clip_224.openai"  # OpenAI CLIP ViT-B/16 (saf CLIP; in1k-ft DEĞİL)
PATCH_SIZE = 16
OUT_CHANNELS = 256

# Veri hattının uyguladığı ImageNet norm (datasets/transforms.py ile aynı) ve CLIP'in
# kendi norm'u. trunk girişini ImageNet-norm'dan CLIP-norm'a çevirmek için kullanılır.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _out_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bir piramit seviyesini `out_ch` kanala indiren lateral(1x1)+output(3x3) bloğu
    (dino_backbone._out_conv ile aynı; GroupNorm'lu, sıfırdan eğitilen neck için)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
    )


class ClipBackbone(nn.Module):
    """CLIP ViT-B/16 + Simple Feature Pyramid, BackboneWithFPN ile aynı arayüz.

    forward(images) -> OrderedDict {"0","1","2","3","pool"} (strides 4/8/16/32/64,
    her biri out_channels kanal). RetinaNet + seg/cls head'leri değişmeden çalışır.

    trainable_blocks: DINO ile aynı anlam. 0 = ViT gövdesi tamamen donuk (kanonik;
        sadece neck + head'ler eğitilir), N = son N blok + final norm, 12 = tüm ViT.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
    ):
        super().__init__()
        self.vit = timm.create_model(
            CLIP_MODEL,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,  # 224 dışı girişlerde pos-embed interpolasyonu
        )
        self.num_prefix_tokens = self.vit.num_prefix_tokens  # CLS token(lar)ı atmak için
        embed_dim = self.vit.embed_dim  # 768
        self.out_channels = out_channels

        self._set_trainable_blocks(trainable_blocks)

        # ImageNet-norm -> CLIP-norm yeniden ölçekleme sabitleri (buffer -> doğru cihaz/dtype).
        # x_clip = (x_in * imagenet_std + imagenet_mean - clip_mean) / clip_std
        #        = x_in * scale + shift
        im_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        im_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        cl_std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        cl_mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self.register_buffer("norm_scale", im_std / cl_std)
        self.register_buffer("norm_shift", (im_mean - cl_mean) / cl_std)

        # --- Simple Feature Pyramid (dino_backbone.py ile birebir aynı) ---
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(32, embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
        )
        self.out0 = _out_conv(embed_dim // 4, out_channels)
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.out1 = _out_conv(embed_dim // 2, out_channels)
        self.out2 = _out_conv(embed_dim, out_channels)
        self.out3 = _out_conv(embed_dim, out_channels)

    def _set_trainable_blocks(self, trainable_blocks: int) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        if trainable_blocks and trainable_blocks > 0:
            blocks = self.vit.blocks
            n = min(trainable_blocks, len(blocks))
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.vit.norm.parameters():
                p.requires_grad = True

    def _tokens_to_grid(self, images: Tensor) -> Tensor:
        """ViT patch token'larını (B, embed, h, w) uzamsal haritaya çevirir."""
        b, _, h_img, w_img = images.shape
        tokens = self.vit.forward_features(images)  # (B, prefix+N, embed), norm uygulanmış
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]  # CLS'i at -> (B, N, embed)
        h, w = h_img // PATCH_SIZE, w_img // PATCH_SIZE
        return patch_tokens.transpose(1, 2).reshape(b, -1, h, w)  # (B, embed, h, w)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK ViT gövdesinin çıktısı: (B, embed, h, w) grid. Feature-caching için ayrıldı
        (dino_backbone.trunk_forward ile aynı gerekçe: donukken deterministik -> cache'lenebilir).

        Girişi önce ImageNet-norm'dan CLIP-norm'a çevirir (bkz. modül docstring)."""
        images = images * self.norm_scale + self.norm_shift  # ImageNet-norm -> CLIP-norm
        return self._tokens_to_grid(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        """EĞİTİLEBİLİR neck (Simple Feature Pyramid): trunk grid -> 5-seviye piramit."""
        p0 = self.out0(self.up4(x))                               # stride 4
        p1 = self.out1(self.up2(x))                               # stride 8
        p2 = self.out2(x)                                         # stride 16
        p3 = self.out3(F.max_pool2d(x, kernel_size=2, stride=2))  # stride 32
        pool = F.max_pool2d(p3, kernel_size=1, stride=2, padding=0)  # stride 64
        return OrderedDict([("0", p0), ("1", p1), ("2", p2), ("3", p3), ("pool", pool)])

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.neck_forward(self.trunk_forward(images))
