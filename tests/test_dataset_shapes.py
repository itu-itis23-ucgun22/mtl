"""Shape/dtype regression check for CocoMultiTaskDataset, against a tiny
synthetic COCO-format fixture built on the fly (no committed binary files)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mtl.datasets.coco_multitask import CocoMultiTaskDataset

IMG_SIZE = 64


def _make_fixture(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    Image.fromarray((np.random.rand(100, 80, 3) * 255).astype(np.uint8)).save(img_dir / "000001.jpg")

    coco_json = {
        "images": [{"id": 1, "file_name": "000001.jpg", "width": 80, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 30],
                "area": 600,
                "iscrowd": 0,
                "segmentation": [[10, 10, 30, 10, 30, 40, 10, 40]],
            }
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    }
    ann_path = ann_dir / "instances.json"
    ann_path.write_text(json.dumps(coco_json))
    return ann_path, img_dir


def test_dataset_output_shapes(tmp_path):
    ann_path, img_dir = _make_fixture(tmp_path)
    dataset = CocoMultiTaskDataset(str(ann_path), str(img_dir), img_size=IMG_SIZE, train=False)

    assert len(dataset) == 1
    image, target = dataset[0]

    assert image.shape == (3, IMG_SIZE, IMG_SIZE)
    assert target["boxes"].dtype == torch.float32
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].dtype == torch.int64
    assert 0 <= target["labels"].item() < dataset.num_classes
    assert target["sem_mask"].shape == (IMG_SIZE, IMG_SIZE)
    assert target["sem_mask"].dtype == torch.int64
    assert target["cls_labels"].shape == (dataset.num_classes,)
    assert set(target["cls_labels"].unique().tolist()) <= {0.0, 1.0}
