"""Takılabilir omurga (gövde + neck), her üç head tarafından paylaşılır.

İki seçenek de aynı çıktı sözleşmesini üretir: forward(images) -> OrderedDict
{"0","1","2","3","pool"} (strides 4/8/16/32/64), hepsi 256 kanal ve modülde
`.out_channels = 256`. Bu sözleşme ResNet'e özgü DEĞİL - FPN yalnızca onu üreten
bir neck. Head'ler neck'in çıktısını tüketir, hangi neck olduğunu umursamaz.

  - "resnet50": gövde ResNet50 + neck FPN (torchvision `resnet_fpn_backbone`).
  - "dino_vitb16": gövde DINOv1 ViT-B/16 + neck Simple Feature Pyramid
    (bkz. models/dino_backbone.py; gerekçe EXPERIMENTS.md "Kararlar" notu).
"""
from __future__ import annotations

from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights
from torchvision.ops import misc as misc_nn_ops


def build_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
    trainable_layers: int = 3,
) -> nn.Module:
    """`.out_channels=256` özniteliği olan ve 5-seviyeli feature dict döndüren bir omurga kur.

    trainable_layers'ın anlamı omurgaya göre değişir:
      - resnet50: eğitilebilir ResNet katmanı sayısı. 3 = stem+layer1 donuk, layer2-4
        eğitilebilir (~20k görüntüde ImageNet feature'larını felaketsiz fine-tune etmek
        için makul default). 0 = tüm gövde donuk, 5 = stem dahil hepsi eğitilebilir.
      - dino_vitb16: eğitilebilir transformer bloğu sayısı (sondan). 0 = ViT tamamen
        donuk (kanonik SSL kullanımı; sadece neck+head'ler eğitilir), 12 = tüm ViT.
    """
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return resnet_fpn_backbone(
            backbone_name=name,
            weights=weights,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            trainable_layers=trainable_layers,
        )

    if name in ("dino_vitb16", "dino"):
        # Lazy import: timm sadece DINO omurgası istendiğinde gereksin (ResNet-only
        # koşular ve ortamlar timm'siz de çalışsın).
        from mtl.models.dino_backbone import DinoBackbone

        return DinoBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    raise NotImplementedError(
        f"Unknown backbone '{name}'. Supported: 'resnet50', 'dino_vitb16'."
    )
