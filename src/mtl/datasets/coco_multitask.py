"""CocoMultiTaskDataset: one __getitem__ returns labels for all three tasks.

Category id mapping: COCO category ids are non-contiguous (1..90, gaps).
We map them to contiguous indices 0..num_classes-1, sorted by id, and reuse
that mapping across all three tasks:
  - detection labels: the contiguous index directly (0..num_classes-1),
    since RetinaNet uses per-class sigmoid scores with no explicit
    background class.
  - segmentation mask pixel values: contiguous index + 1 (1..num_classes),
    0 reserved for background, 255 for ignored/unlabeled pixels.
  - classification target: a (num_classes,) multi-hot vector over the same
    contiguous indices.

`iscrowd` annotations are dropped entirely for v1 (both detection and
segmentation) - handling crowd regions correctly (as an "ignore" region
rather than a hard negative) is a well-known but nontrivial detail, noted
here rather than silently done wrong.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data import Dataset

from mtl.datasets.transforms import build_transforms


class CocoMultiTaskDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        img_size: int = 512,
        train: bool = True,
        n_images: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ):
        self.coco = COCO(ann_file)
        self.img_dir = Path(img_dir)
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

        self.img_ids = sorted(self.coco.getImgIds())
        if n_images is not None:
            self.img_ids = self.img_ids[:n_images]

        self.transforms = transforms if transforms is not None else build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_id = self.img_ids[index]
        img_info = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.img_dir / img_info["file_name"]).convert("RGB")
        width, height = image.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        boxes, labels = [], []
        sem_mask = np.zeros((height, width), dtype=np.int64)
        cls_labels = torch.zeros(self.num_classes, dtype=torch.float32)

        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            class_idx = self.cat_id_to_idx[ann["category_id"]]

            boxes.append([x, y, x + w, y + h])
            labels.append(class_idx)
            cls_labels[class_idx] = 1.0

            mask = self.coco.annToMask(ann).astype(bool)
            sem_mask[mask] = class_idx + 1

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "sem_mask": torch.from_numpy(sem_mask),
            "cls_labels": cls_labels,
            "image_id": torch.tensor([image_id]),
        }
        image_t, target = self.transforms(image, target)
        return image_t, target
