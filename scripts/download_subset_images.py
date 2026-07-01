"""CLI: download only the images referenced by a (already-filtered) COCO
subset annotation JSON, using each image's `coco_url` field - much cheaper
than downloading the full train2017/val2017 zip (~18GB / ~1GB) when the
subset is a fraction of the full set.

    python scripts/download_subset_images.py \
        --ann-file data/coco_subset/annotations/instances_train_subset.json \
        --out-dir data/coco_subset/images/train

Skips files that already exist, so it's safe to re-run after a partial/
interrupted download.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


def _download_one(image: dict, out_dir: Path, retries: int, timeout: float) -> str | None:
    dest = out_dir / image["file_name"]
    if dest.exists():
        return None
    url = image.get("coco_url")
    if not url:
        return f"image {image['id']}: no coco_url field"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return None
        except requests.RequestException as exc:
            if attempt == retries - 1:
                return f"image {image['id']} ({url}): {exc}"
            time.sleep(1.0)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True, help="filtered COCO subset JSON (from prepare_coco_subset.py)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = json.loads(Path(args.ann_file).read_text())["images"]

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download_one, img, out_dir, args.retries, args.timeout) for img in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="downloading images"):
            err = future.result()
            if err:
                errors.append(err)

    print(f"Done. {len(images) - len(errors)}/{len(images)} images present in {out_dir}.")
    if errors:
        print(f"{len(errors)} failed, e.g.:")
        for err in errors[:10]:
            print(" ", err)


if __name__ == "__main__":
    main()
