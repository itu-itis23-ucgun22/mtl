"""Joint (image, target) transforms - boxes and the segmentation mask must
be transformed together with the image, so these can't be plain
torchvision.transforms (which only see the image).

Resize distorts aspect ratio (plain resize to a fixed square) rather than
letterboxing, for v1 simplicity - every image in a batch ends up the exact
same shape, which is what lets MultiTaskModel skip RetinaNet's own
variable-size GeneralizedRCNNTransform (see models/detection_head.py).
"""
from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        orig_w, orig_h = image.size
        image = image.resize((self.size, self.size), Image.BILINEAR)

        boxes = target["boxes"]
        if boxes.numel():
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= self.size / orig_w
            boxes[:, [1, 3]] *= self.size / orig_h
        target["boxes"] = boxes

        sem_mask = target["sem_mask"].unsqueeze(0).unsqueeze(0).float()
        sem_mask = F.interpolate(sem_mask, size=(self.size, self.size), mode="nearest")
        target["sem_mask"] = sem_mask[0, 0].long()
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        if random.random() >= self.p:
            return image, target
        w, _ = image.size
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = target["boxes"]
        if boxes.numel():
            boxes = boxes.clone()
            x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
        target["boxes"] = boxes
        target["sem_mask"] = target["sem_mask"].flip(-1)
        return image, target


class ToTensorNormalize:
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Tensor, Dict]:
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, target


def build_transforms(img_size: int, train: bool) -> Compose:
    ts = [Resize(img_size)]
    if train:
        ts.append(RandomHorizontalFlip())
    ts.append(ToTensorNormalize())
    return Compose(ts)
