"""Takılabilir omurga (gövde + neck), her üç head tarafından paylaşılır.

İki seçenek de aynı çıktı sözleşmesini üretir: forward(images) -> OrderedDict
{"0","1","2","3","pool"} (strides 4/8/16/32/64), hepsi 256 kanal ve modülde
`.out_channels = 256`. Bu sözleşme ResNet'e özgü DEĞİL - FPN yalnızca onu üreten
bir neck. Head'ler neck'in çıktısını tüketir, hangi neck olduğunu umursamaz.

  - "resnet50": gövde ResNet50 + neck FPN (torchvision `resnet_fpn_backbone`).
  - "dino_vitb16": gövde DINOv1 ViT-B/16 + neck Simple Feature Pyramid (models/dino_backbone.py).
  - "clip_vitb16": gövde CLIP ViT-B/16 (OpenAI) + neck Simple Feature Pyramid (models/clip_backbone.py;
    DINOv1 ile aynı patch16 -> aynı grid; DINOv1 baseline'ı bozmamak için ayrı dosya).
  - "mae_vitb16": gövde MAE ViT-B/16 (maskeli piksel yeniden-kurma / MIM) + Simple Feature Pyramid
    (models/mae_backbone.py; DINOv1/CLIP/SAM ile TAM adil: aynı ViT-B, patch16, 32x32 grid).
  - "deit_vitb16": gövde DeiT ViT-B/16 (SUPERVISED in1k) + Simple Feature Pyramid
    (models/deit_backbone.py; supervised paradigmanın ADİL ViT temsilcisi -> ResNet'in conv/FPN
    confound'unu kapatır).
  - "sam_vitb16": gövde SAM image encoder (timm samvit_base_patch16) + Simple Feature Pyramid
    (models/sam_backbone.py; segmentation-native paradigma).
  - "ijepa_vith14": gövde I-JEPA ViT-H/14 (HF transformers) + Simple Feature Pyramid
    (models/ijepa_backbone.py; predictive-SSL paradigma; ViT-H -> boyut confound, bkz. dosya notu).
  - "dinov2_vitb14"(_reg) / "dinov2_vits14"(_reg): gövde DINOv2 ViT + Simple Feature Pyramid
    (models/dinov2_backbone.py; DINOv1 baseline'ı bozmamak için ayrı dosya).
  (gerekçe EXPERIMENTS.md "Kararlar" notu).
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
      - dino* (DINOv1/DINOv2): eğitilebilir transformer bloğu sayısı (sondan). 0 = ViT tamamen
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

    # Lazy import'lar: timm sadece DINO omurgası istendiğinde gereksin (ResNet-only koşular
    # ve ortamlar timm'siz de çalışsın). DINOv2 kendi dosyasında; DINOv1 baseline'ı bozulmaz.
    if name.startswith("dinov2"):
        from mtl.models.dinov2_backbone import Dinov2Backbone

        return Dinov2Backbone(model_name=name, pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("dino_vitb16", "dino"):
        from mtl.models.dino_backbone import DinoBackbone

        return DinoBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("clip_vitb16", "clip"):
        from mtl.models.clip_backbone import ClipBackbone

        return ClipBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("mae_vitb16", "mae"):
        from mtl.models.mae_backbone import MaeBackbone

        return MaeBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("deit_vitb16", "deit"):
        from mtl.models.deit_backbone import DeitBackbone

        return DeitBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("moco_vitb16", "moco", "mocov3"):
        from mtl.models.moco_backbone import MocoBackbone

        return MocoBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("beit_vitb16", "beit"):
        from mtl.models.beit_backbone import BeitBackbone

        return BeitBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("sam_vitb16", "sam"):
        from mtl.models.sam_backbone import SamBackbone

        return SamBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    if name in ("ijepa_vith14", "ijepa"):
        from mtl.models.ijepa_backbone import IjepaBackbone

        return IjepaBackbone(pretrained=pretrained, trainable_blocks=trainable_layers)

    raise NotImplementedError(
        f"Unknown backbone '{name}'. Supported: 'resnet50', 'dino_vitb16', 'clip_vitb16', "
        "'mae_vitb16', 'deit_vitb16', 'moco_vitb16', 'beit_vitb16', 'sam_vitb16', 'ijepa_vith14', "
        "'dinov2_vitb14', 'dinov2_vitb14_reg', 'dinov2_vits14', 'dinov2_vits14_reg'."
    )
