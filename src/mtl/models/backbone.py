"""Shared ResNet50+FPN backbone, reused by all three task heads.

Thin wrapper around torchvision's `resnet_fpn_backbone` so the rest of the
codebase depends on `build_backbone` rather than on torchvision's exact
constructor signature.
"""
from __future__ import annotations

from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights
from torchvision.ops import misc as misc_nn_ops


def build_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
    trainable_layers: int = 0,
) -> BackboneWithFPN:
    """Build a ResNet+FPN backbone.

    Returns a BackboneWithFPN whose forward(images) -> OrderedDict of feature
    maps keyed "0".."3" (strides 4/8/16/32, i.e. FPN levels P2-P5), all with
    out_channels=256.

    trainable_layers=3 freezes the stem + layer1 and trains layer2-4, a
    reasonable default for fine-tuning ImageNet features on ~20k images
    without catastrophic forgetting. trainable_layers=0 freezes all layers, and trainable_layers=5 trains all layers (including the stem). See
    torchvision's `resnet_fpn_backbone` for details.
    """
    if name != "resnet50":
        raise NotImplementedError(f"Only resnet50 is wired up for now, got '{name}'")

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return resnet_fpn_backbone(
        backbone_name=name, 
        weights=weights,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=trainable_layers,
    )
