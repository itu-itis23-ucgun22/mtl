"""DINO (self-supervised ViT-B/16) gövdesi + ViTDet tarzı Simple Feature Pyramid
neck'i - ResNet+FPN omurgasıyla *aynı çıktı sözleşmesini* üretir.

Neden bu yapı (ayrıntılı gerekçe EXPERIMENTS.md "Kararlar" notunda):
  - Head'ler (detection/segmentation/classification) bir neck'in çıktı sözleşmesini
    tüketir: OrderedDict {"0","1","2","3","pool"}, strides 4/8/16/32/64, hepsi 256 kanal
    ve modülde `.out_channels = 256`. Bu sözleşme ResNet'e özgü değil - FPN yalnızca
    onu üreten bir neck. (bkz. models/backbone.py, models/multitask_model.py)
  - DINO ViT *plain* bir transformer: tek çözünürlük (stride 16) üretir, FPN'in
    birleştireceği doğal bir hiyerarşi yoktur. Bu yüzden FPN yerine ViTDet'in
    (Li et al. 2022) Simple Feature Pyramid'i kullanılıyor: tek stride-16 haritadan
    her seviyeyi bağımsız up/down-sample ile türetir. Daha az parametre, T4-dostu,
    ve neck'i minimal tutarak karşılaştırmayı "DINO vs ResNet feature'ları"na odaklar.

timm neden torch.hub yerine: timm `pretrained=False`'ta ağ olmadan (offline) kurulur -
tests ve scripts/eval.py bu yolu kullanır. torch.hub ise `pretrained=False` olsa bile
repo kodunu indirmeye çalışıp offline eval/testi bozardı. timm ayrıca `dynamic_img_size`
ile 224 dışı girişlerde positional-embedding interpolasyonunu otomatik yapar.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

DINO_MODEL = "vit_base_patch16_224.dino"  # DINOv1 ViT-B/16
PATCH_SIZE = 16
OUT_CHANNELS = 256


def _out_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bir piramit seviyesini `out_ch` kanala indiren lateral(1x1)+output(3x3) bloğu.
    (torchvision FPN'in seviye-başı çıkış conv'unun muadili; GroupNorm ile sıfırdan
    eğitilen bu neck'in kararlılığı için norm eklendi.)"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
    )


class DinoBackbone(nn.Module):
    """DINO ViT-B/16 + Simple Feature Pyramid, BackboneWithFPN ile aynı arayüz.

    forward(images) -> OrderedDict {"0","1","2","3","pool"} (strides 4/8/16/32/64,
    her biri out_channels kanal). RetinaNet + seg/cls head'leri değişmeden çalışır.

    trainable_blocks: ViT gövdesinden kaç transformer bloğunun (sondan) eğitileceği.
        0  = gövde tamamen donuk (kanonik SSL kullanımı, T4'te en hızlı; sadece neck +
             head'ler eğitilir),
        N  = son N blok + final norm eğitilebilir,
        12 = tüm ViT eğitilebilir.
    Neck (feature pyramid) ve head'ler her zaman eğitilebilir - bu bayrak yalnızca ViT
    gövdesini etkiler.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
    ):
        super().__init__()
        self.vit = timm.create_model(
            DINO_MODEL,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,  # 224 dışı girişlerde pos-embed interpolasyonu
        )
        self.num_prefix_tokens = self.vit.num_prefix_tokens  # CLS token(lar)ı atmak için
        embed_dim = self.vit.embed_dim  # 768
        self.out_channels = out_channels

        self._set_trainable_blocks(trainable_blocks)

        # --- Simple Feature Pyramid (DINO'nun neck'i) ---
        # Tek stride-16 haritadan 5 seviye; up/down-sample + seviye-başı çıkış conv'u.
        # "0" stride 4: 4x upsample (iki ConvTranspose2d)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(32, embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
        )
        self.out0 = _out_conv(embed_dim // 4, out_channels)
        # "1" stride 8: 2x upsample
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.out1 = _out_conv(embed_dim // 2, out_channels)
        # "2" stride 16: identity ölçek
        self.out2 = _out_conv(embed_dim, out_channels)
        # "3" stride 32: 2x downsample (maxpool, parametresiz - ViTDet gibi)
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

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        x = self._tokens_to_grid(images)  # stride 16
        p0 = self.out0(self.up4(x))                               # stride 4
        p1 = self.out1(self.up2(x))                               # stride 8
        p2 = self.out2(x)                                         # stride 16
        p3 = self.out3(F.max_pool2d(x, kernel_size=2, stride=2))  # stride 32
        # "pool": torchvision LastLevelMaxPool ile aynı (stride 64) - RetinaNet 5 seviye ister
        pool = F.max_pool2d(p3, kernel_size=1, stride=2, padding=0)
        return OrderedDict([("0", p0), ("1", p1), ("2", p2), ("3", p3), ("pool", pool)])
