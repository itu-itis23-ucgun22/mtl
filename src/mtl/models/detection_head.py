"""Detection head: builds a torchvision RetinaNet around our shared backbone.

We construct a full `RetinaNet` instance (so we get its default anchor
generator, head, and loss/postprocess logic for free), but `MultiTaskModel`
never calls `RetinaNet.forward` directly. Instead it drives
`retinanet.head`, `retinanet.anchor_generator`, and `retinanet.compute_loss`
by hand off of a feature dict computed once and shared with the other two
heads (see multitask_model.py).

Implementation note vs. the original plan: RetinaNet's own `.transform`
submodule (GeneralizedRCNNTransform) is built for *variable*-sized inputs
and re-normalizes images internally. Our dataset already resizes every
image to a fixed `img_size` square and normalizes with ImageNet mean/std
(datasets/transforms.py), so every image in a batch has identical shape.
Given that, we skip `.transform` entirely and build a plain
`torchvision.models.detection.image_list.ImageList` directly from the
already-batched, already-normalized tensor - simpler and avoids double
normalization, while still reusing RetinaNet's head/anchors/loss verbatim.
"""
from __future__ import annotations

from torch import nn
from torchvision.models.detection import RetinaNet


def build_detection_model(backbone: nn.Module, num_classes: int) -> RetinaNet:
    """num_classes = number of foreground COCO categories (80); RetinaNet's
    per-class sigmoid classification needs no separate background class.

    RetinaNet yalnızca `backbone.out_channels` + 5-seviyeli feature dict döndüren
    bir forward bekler; illa `BackboneWithFPN` olması gerekmez. Bu sayede ResNet+FPN
    veya DINO+SimpleFeaturePyramid (models/dino_backbone.py) aynı şekilde takılabilir.
    """
    return RetinaNet(backbone, num_classes=num_classes)
