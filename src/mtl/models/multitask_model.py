"""The combined multi-task model: one shared backbone + three task heads.

Runs the shared ResNet50+FPN backbone exactly once per batch and feeds the
resulting feature dict into all three heads, instead of calling
`RetinaNet.forward` (which would run the backbone again internally).

Label convention (see datasets/coco_multitask.py for the category mapping):
  - detection labels: contiguous 0..num_classes-1 (RetinaNet has no explicit
    background class - it uses independent per-class sigmoid scores).
  - segmentation mask pixel values: 0 = background, 1..num_classes = the
    same contiguous category indices + 1, 255 = ignore.
  - classification target: a (num_classes,) multi-hot float vector using
    the same contiguous indices as detection.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.detection.image_list import ImageList

from mtl.models.backbone import build_backbone
from mtl.models.classification_head import MultiLabelClsHead
from mtl.models.detection_head import build_detection_model
from mtl.models.segmentation_head import SemanticSegHead

FPN_OUT_CHANNELS = 256


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        det_num_classes: int = 80,
        seg_num_classes: int = 81,  # +1 for background
        cls_num_labels: int = 80,
        lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_targets: str = "qkv,proj",
        lora_blocks: int = -1,
        adaptive_loss: bool = False,   # Faz 3: öğrenilen belirsizlik ağırlıkları (Kendall 2018)
        seg_neck: str = "fcn",         # Faz 3: seg decoder "fcn" | "aspp" (görev-özel neck)
    ):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained, trainable_backbone_layers)
        if lora:
            # ViT gövdesine LoRA adaptörleri tak (taban donuk kalır). Cache KULLANILAMAZ →
            # normal scripts/train.py ile koş (bkz. models/lora.py, ROADMAP Faz 2).
            from mtl.models.lora import apply_lora_to_vit

            if not hasattr(self.backbone, "vit"):
                raise ValueError(
                    f"LoRA yalnızca ViT gövdeli backbone'larda desteklenir (dino/dinov2/mae/... '.vit'); "
                    f"'{backbone_name}' uygun değil."
                )
            n = apply_lora_to_vit(
                self.backbone.vit,
                rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
                targets=tuple(t.strip() for t in lora_targets.split(",") if t.strip()),
                last_n_blocks=lora_blocks,
            )
            if n == 0:
                raise ValueError(
                    f"LoRA hiçbir Linear'a uygulanmadı (targets={lora_targets}, blocks={lora_blocks}) "
                    f"— '{backbone_name}' gövde yapısı beklenenden farklı olabilir."
                )
            print(f"[lora] {n} Linear'a LoRA enjekte edildi "
                  f"(rank={lora_rank}, alpha={lora_alpha}, targets={lora_targets}, blocks={lora_blocks})")
        self.detection_model = build_detection_model(self.backbone, det_num_classes)
        self.seg_head = SemanticSegHead(FPN_OUT_CHANNELS, seg_num_classes, neck=seg_neck)
        self.cls_head = MultiLabelClsHead(FPN_OUT_CHANNELS, cls_num_labels)
        # Faz 3: adaptif loss ağırlıklandırıcı (alt-modül → params optimizer'a + checkpoint'e otomatik girer)
        self.loss_weighter = None
        if adaptive_loss:
            from mtl.losses.joint_loss import UncertaintyWeighter
            self.loss_weighter = UncertaintyWeighter()

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        if self.training and targets is None:
            raise ValueError("targets must be provided in training mode")
        _, _, height, width = images.shape
        features = self.backbone(images)  # OrderedDict: "0".."3" (+ "pool")
        return self._run_heads(features, (height, width), images.device, targets)

    def forward_from_trunk(
        self, trunk: Tensor, image_hw: tuple, targets: Optional[List[Dict[str, Tensor]]] = None
    ):
        """Önceden hesaplanmış DONUK backbone trunk çıktısından neck+head koşar; backbone atlanır.

        trunk: `backbone.trunk_forward(images)` çıktısı, (B, embed, h, w). image_hw: yeniden
        boyutlanmış görüntü boyutu (H, W) - anchor stride'ı ve seg upsample için gerekli.
        Yalnızca `neck_forward`'ı olan backbone'larda (DINO ailesi) geçerli. Feature-caching
        eğitimi bunu kullanır (scripts/train_cached.py): pahalı ViT forward'ı atlanır.
        """
        if self.training and targets is None:
            raise ValueError("targets must be provided in training mode")
        features = self.backbone.neck_forward(trunk)
        return self._run_heads(features, image_hw, trunk.device, targets)

    def _run_heads(self, features, image_hw, device, targets):
        """features (5-seviye piramit) -> üç head. forward ve forward_from_trunk'ın ortak yolu."""
        height, width = image_hw
        features_list = list(features.values())
        batch_size = features_list[0].shape[0]
        image_sizes = [(height, width)] * batch_size

        # Anchor generator yalnızca image_list.tensors.shape[-2:]'i okur (dtype/device feature'dan
        # gelir); gerçek görüntü içeriği gerekmez, o yüzden hafif bir dummy yeter - bu sayede
        # cache'ten (görüntüsüz) çalışırken de anchor'lar aynı üretilir.
        image_list = ImageList(torch.empty((batch_size, 1, height, width), device=device), image_sizes)
        anchors = self.detection_model.anchor_generator(image_list, features_list)
        det_head_outputs = self.detection_model.head(features_list)

        seg_logits = self.seg_head(features, output_size=(height, width))
        cls_logits = self.cls_head(features)

        if self.training:
            det_targets = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            det_losses = self.detection_model.compute_loss(det_targets, det_head_outputs, anchors)

            sem_masks = torch.stack([t["sem_mask"] for t in targets])
            seg_loss = F.cross_entropy(seg_logits, sem_masks, ignore_index=255)

            cls_targets = torch.stack([t["cls_labels"] for t in targets])
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)

            return {
                "classification": det_losses["classification"],
                "bbox_regression": det_losses["bbox_regression"],
                "seg_loss": seg_loss,
                "cls_loss": cls_loss,
            }

        # RetinaNet.postprocess_detections expects head_outputs/anchors split
        # per FPN level (this is normally done inside RetinaNet.forward,
        # which we bypass - see module docstring / detection_head.py).
        num_anchors_per_level = [f.shape[2] * f.shape[3] for f in features_list]
        anchors_per_loc = det_head_outputs["cls_logits"].size(1) // sum(num_anchors_per_level)
        num_anchors_per_level = [n * anchors_per_loc for n in num_anchors_per_level]

        split_head_outputs = {k: list(v.split(num_anchors_per_level, dim=1)) for k, v in det_head_outputs.items()}
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        detections = self.detection_model.postprocess_detections(split_head_outputs, split_anchors, image_sizes)
        return {
            "detections": detections,
            "seg_pred": seg_logits.argmax(dim=1),
            "cls_pred": torch.sigmoid(cls_logits),
        }
