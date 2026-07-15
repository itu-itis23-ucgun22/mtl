"""CLI: build the filtered ~20-25k image COCO subset used for training/val.

    python scripts/prepare_coco_subset.py \
        --ann-file /path/to/annotations/instances_train2017.json \
        --out data/coco_subset/annotations/instances_train_subset.json \
        --n-images 22500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtl.datasets.coco_subset import build_coco_subset_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True, help="path to full COCO instances_*.json")
    parser.add_argument("--out", required=True, help="output path for the filtered subset JSON")
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-instances-per-image", type=int, default=40)
    parser.add_argument("--min-images-per-category", type=int, default=20)
    parser.add_argument(
        "--exclude", default=None,
        help="mevcut bir subset JSON'ı; bu görüntüler seçim havuzundan ÇIKARILIR "
             "(ör. test setini val subset'iyle çakışmasın diye: --exclude .../instances_val_subset.json)",
    )
    args = parser.parse_args()

    exclude_ids = None
    if args.exclude:
        data = json.loads(Path(args.exclude).read_text())
        exclude_ids = [img["id"] for img in data["images"]]
        print(f"{len(exclude_ids)} görüntü hariç tutuluyor ({args.exclude})")

    n_selected = build_coco_subset_index(
        ann_file=args.ann_file,
        out_json=args.out,
        n_images=args.n_images,
        seed=args.seed,
        max_instances_per_image=args.max_instances_per_image,
        min_images_per_category=args.min_images_per_category,
        exclude_ids=exclude_ids,
    )
    print(f"Wrote {n_selected} images to {args.out}")


if __name__ == "__main__":
    main()
