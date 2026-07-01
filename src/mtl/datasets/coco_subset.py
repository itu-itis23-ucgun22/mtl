"""Select a ~20-25k image COCO subset and write it out as a standalone
COCO-format annotation file, so downstream code keeps using the standard
`pycocotools.coco.COCO` loader unmodified.

Called from scripts/prepare_coco_subset.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from pycocotools.coco import COCO


def build_coco_subset_index(
    ann_file: str,
    out_json: str,
    n_images: int,
    seed: int = 42,
    max_instances_per_image: int = 40,
    min_images_per_category: int = 20,
) -> int:
    """Sample ~n_images images (all 80 categories kept) with at least one
    annotation and at most max_instances_per_image annotations, biased to
    guarantee every category gets at least min_images_per_category images
    before filling the rest randomly - so rare categories aren't lost.

    Writes a filtered COCO-format JSON to out_json. Returns the number of
    images selected.
    """
    coco = COCO(ann_file)
    cat_ids = sorted(coco.getCatIds())

    candidate_ids = set()
    for img_id in coco.getImgIds():
        n_ann = len(coco.getAnnIds(imgIds=img_id))
        if 0 < n_ann <= max_instances_per_image:
            candidate_ids.add(img_id)

    rng = random.Random(seed)
    cat_to_imgs = {cat_id: [i for i in coco.getImgIds(catIds=[cat_id]) if i in candidate_ids] for cat_id in cat_ids}

    selected: set[int] = set()
    for cat_id in sorted(cat_ids, key=lambda c: len(cat_to_imgs[c])):
        imgs = cat_to_imgs[cat_id][:]
        rng.shuffle(imgs)
        selected.update(imgs[:min_images_per_category])
        if len(selected) >= n_images:
            break

    remaining = [i for i in candidate_ids if i not in selected]
    rng.shuffle(remaining)
    for img_id in remaining:
        if len(selected) >= n_images:
            break
        selected.add(img_id)

    selected_ids = list(selected)[:n_images]

    subset = {
        "images": coco.loadImgs(selected_ids),
        "annotations": coco.loadAnns(coco.getAnnIds(imgIds=selected_ids)),
        "categories": coco.loadCats(cat_ids),
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset))
    return len(selected_ids)
