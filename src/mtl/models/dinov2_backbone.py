"""DINOv2 (self-supervised ViT, patch14) gövdesi + ViTDet tarzı Simple Feature Pyramid.

DINOv1 omurgasıyla (models/dino_backbone.py) *aynı çıktı sözleşmesini* üretir: forward(images)
-> OrderedDict {"0","1","2","3","pool"} (256 kanal, `.out_channels = 256`), böylece ResNet /
DINOv1 ile birebir aynı head'ler + RetinaNet üzerinde takılır. Gerekçe: EXPERIMENTS.md "Kararlar".

DINOv1'den ayrı dosyada tutulmasının nedeni: DINOv1 baseline'ı (mevcut sonuç/checkpoint'ler ona
bağlı) hiç değiştirmeden bırakmak. Ortak olan tek parça neck'in seviye-başı çıkış conv'u
(`_out_conv`) - onu import ediyoruz; DINOv1'e özgü patch16/model sabitleri burada tekrar etmez.

DINOv1'e göre farklar:
  - patch16 yerine **patch14** -> girişin 14'e bölünebilmesi beklenir (config'te img_size=518;
    aksi halde timm patch-embed conv'u kenardan birkaç px kırpar, patlamaz ama gridi kaydırır).
  - "stride 4/8/16/32/64" etiketleri artık patch14 tabanlı nominal değerlerdir (gerçekte ~3.5x
    katları); RetinaNet anchor'ları feature-map boyutundan türediği ve seg head tam çözünürlüğe
    upsample ettiği için bu fark head'leri etkilemez.
  - register'lı ("_reg") varyantlarda CLS dışında 4 register token daha vardır; `num_prefix_tokens`
    gövdeden dinamik okunduğu için ekstra kod gerekmez.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mtl.models.dino_backbone import OUT_CHANNELS, _out_conv

PATCH_SIZE = 14  # DINOv2 tüm boyutlarda patch14

# İsim -> timm model adı. Yeni bir DINOv2 boyutu eklemek buraya tek satır (embed_dim ViT-S/B/L
# ve register token sayısı gövdeden dinamik okunur, başka kod değişmez).
DINOV2_MODELS: Dict[str, str] = {
    "dinov2_vits14":     "vit_small_patch14_dinov2.lvd142m",       # ViT-S/14 (384-dim, hafif)
    "dinov2_vitb14":     "vit_base_patch14_dinov2.lvd142m",        # ViT-B/14 (768-dim)
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",  # ViT-S/14 + 4 register
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",   # ViT-B/14 + 4 register
}

DEFAULT_MODEL = "dinov2_vitb14_reg"  # register'lı ViT-B: genelde en stabil, temiz attention


class Dinov2Backbone(nn.Module):
    """DINOv2 ViT (patch14) + Simple Feature Pyramid, DinoBackbone ile aynı arayüz.

    forward(images) -> OrderedDict {"0","1","2","3","pool"} (256 kanal). RetinaNet + seg/cls
    head'leri değişmeden çalışır.

    model_name: DINOV2_MODELS anahtarlarından biri (vits14 / vitb14, opsiyonel "_reg").
    trainable_blocks: ViT gövdesinden kaç transformer bloğunun (sondan) eğitileceği.
        0  = gövde tamamen donuk (kanonik SSL kullanımı, en hızlı; sadece neck + head'ler),
        N  = son N blok + final norm eğitilebilir,
        12 = tüm ViT eğitilebilir.
    Neck ve head'ler her zaman eğitilebilir - bu bayrak yalnızca ViT gövdesini etkiler.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
    ):
        super().__init__()
        if model_name not in DINOV2_MODELS:
            raise ValueError(
                f"Unknown DINOv2 backbone '{model_name}'. Options: {sorted(DINOV2_MODELS)}"
            )
        self.patch_size = PATCH_SIZE
        self.vit = timm.create_model(
            DINOV2_MODELS[model_name],
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,  # native (518) dışı girişlerde pos-embed interpolasyonu
        )
        self.num_prefix_tokens = self.vit.num_prefix_tokens  # CLS + register token(lar)ı atmak için
        embed_dim = self.vit.embed_dim  # ViT-S: 384, ViT-B: 768
        self.out_channels = out_channels

        self._set_trainable_blocks(trainable_blocks)

        # --- Simple Feature Pyramid (DINOv1 ile aynı yapı) ---
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
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]  # CLS+register'ı at -> (B, N, embed)
        h, w = h_img // self.patch_size, w_img // self.patch_size
        return patch_tokens.transpose(1, 2).reshape(b, -1, h, w)  # (B, embed, h, w)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK ViT gövdesinin çıktısı: (B, embed, h, w) grid. Feature-caching için ayrıldı
        (bkz. dino_backbone.py trunk_forward + scripts/precompute_features.py)."""
        return self._tokens_to_grid(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        """EĞİTİLEBİLİR neck (Simple Feature Pyramid): trunk grid -> 5-seviye piramit."""
        p0 = self.out0(self.up4(x))                               # stride 4
        p1 = self.out1(self.up2(x))                               # stride 8
        p2 = self.out2(x)                                         # stride 16
        p3 = self.out3(F.max_pool2d(x, kernel_size=2, stride=2))  # stride 32
        # "pool": torchvision LastLevelMaxPool ile aynı (stride 64) - RetinaNet 5 seviye ister
        pool = F.max_pool2d(p3, kernel_size=1, stride=2, padding=0)
        return OrderedDict([("0", p0), ("1", p1), ("2", p2), ("3", p3), ("pool", pool)])

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.neck_forward(self.trunk_forward(images))
