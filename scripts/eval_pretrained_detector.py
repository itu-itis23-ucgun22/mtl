"""REFERANS: COCO-pretrained detektörü (torchvision) bizim val'de ZERO-SHOT değerlendirir.

Multi-task detection'ımızla KIYAS için bir referans üretir: aynı val subset, aynı metrik
(pycocotools bbox mAP → parite bizim engine/evaluate.py ile). Bu "full-COCO'da eğitilmiş SOTA
detektör tavanı"dır — ⚠️ KIYAS DEĞİL, REFERANS: zero-shot (hiç eğitilmedi) + FULL COCO verisi
(118k) gördü, biz 22.5k subset + frozen. RESULTS'a bu rejim-etiketiyle işlenir.

Sınıf eşlemesi: torchvision COCO detection modellerinin çıktı label'ları COCO category_id'lerdir
(1..90, boşluklu) → bizim val JSON'ının category_id'leriyle DOĞRUDAN hizalı (ekstra eşleme gerekmez).

    python scripts/eval_pretrained_detector.py --config configs/train_colab_gpu.yaml
    python scripts/eval_pretrained_detector.py --config configs/train_colab_gpu.yaml --model retinanet
    # test split için: --ann-file .../instances_test_subset.json --img-dir .../images/test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms.functional import to_tensor

from mtl.config import load_config
from mtl.utils.device import resolve_device
from mtl.utils.results import append_result


def build_detector(name: str):
    """torchvision COCO-pretrained detektör (weights='DEFAULT' → en iyi COCO checkpoint)."""
    det = torchvision.models.detection
    if name == "fasterrcnn":
        return det.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    if name == "retinanet":
        return det.retinanet_resnet50_fpn_v2(weights="DEFAULT")
    if name == "fcos":
        return det.fcos_resnet50_fpn(weights="DEFAULT")
    raise ValueError(f"Bilinmeyen model '{name}'. Seçenekler: fasterrcnn, retinanet, fcos.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="val yollarını okumak için (model config'i değil)")
    parser.add_argument("--model", default="fasterrcnn", choices=["fasterrcnn", "retinanet", "fcos"])
    parser.add_argument("--ann-file", default=None, help="verilmezse config'in val'i")
    parser.add_argument("--img-dir", default=None)
    parser.add_argument("--results-csv", default="runs/results.csv")
    parser.add_argument("--score-thresh", type=float, default=0.05, help="COCOeval hızı için düşük eşik")
    parser.add_argument("--split-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    ann_file = args.ann_file or cfg.data.val_ann_file
    img_dir = Path(args.img_dir or cfg.data.val_img_dir)

    coco = COCO(ann_file)
    model = build_detector(args.model).eval().to(device)

    detections = []
    img_ids = coco.getImgIds()
    with torch.no_grad():
        for n, img_id in enumerate(img_ids, 1):
            info = coco.loadImgs(img_id)[0]
            image = Image.open(img_dir / info["file_name"]).convert("RGB")
            x = to_tensor(image).to(device)  # [0,1], native çözünürlük; model kendi resize+norm'unu yapar
            out = model([x])[0]
            for box, label, score in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
                s = float(score)
                if s < args.score_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "image_id": img_id,
                    "category_id": int(label),  # torchvision label == COCO category_id
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": s,
                })
            if n % 200 == 0:
                print(f"  {n}/{len(img_ids)} görüntü işlendi")

    if not detections:
        raise SystemExit("Hiç tespit üretilmedi — model/görüntü yollarını kontrol et.")

    coco_dt = coco.loadRes(detections)
    ev = COCOeval(coco, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    mAP = float(ev.stats[0])  # AP@[.50:.95] — bizim detection_mAP ile aynı tanım

    print(f"\n=== REFERANS: pretrained {args.model} (COCO, ZERO-SHOT) — bizim val'de ===")
    print(f"  detection_mAP : {mAP:.4f}   (⚠️ full-COCO tavanı; kıyas değil, referans)")
    append_result(args.results_csv, {
        "run_name": f"ref_{args.model}_coco_zeroshot" + (f"_{args.split_name}" if args.split_name else ""),
        "backbone": f"{args.model}_r50_coco",
        "trainable_layers": "zeroshot",
        "checkpoint": "-",
        "split": args.split_name or ("test" if args.ann_file else "val"),
        "step": 0,
        "detection_mAP": mAP,
    })
    print(f"[results] {args.results_csv}'ye eklendi.")


if __name__ == "__main__":
    main()
