"""Per-task evaluation: detection mAP (pycocotools COCOeval), segmentation
mIoU, classification mAP/F1 (sklearn), all from one pass over the val set.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from mtl.datasets.coco_multitask import CocoMultiTaskDataset


@torch.no_grad()
def evaluate(model: nn.Module, dataset: CocoMultiTaskDataset, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    coco_detections = []
    intersection = np.zeros(dataset.num_classes + 1, dtype=np.int64)
    union = np.zeros(dataset.num_classes + 1, dtype=np.int64)
    all_cls_pred, all_cls_true = [], [] # listelere şu yüzden ihtiyacımız var: eğitim yaparken batchler halinde çalışıyoruz ve her batchin çıktısını tek tek alıp birleştirmemiz gerekiyor. Bu yüzden listelere atıyoruz ve en sonunda np.stack ile birleştiriyoruz.

    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)

        for i, target in enumerate(targets):
            image_id = int(target["image_id"].item())
            img_info = dataset.coco.loadImgs(image_id)[0]
            orig_w, orig_h = img_info["width"], img_info["height"]
            sx, sy = orig_w / images.shape[-1], orig_h / images.shape[-2]

            det = outputs["detections"][i]
            for box, score, label in zip(det["boxes"].cpu(), det["scores"].cpu(), det["labels"].cpu()):
                x1, y1, x2, y2 = box.tolist()
                x1, x2 = x1 * sx, x2 * sx
                y1, y2 = y1 * sy, y2 * sy
                coco_detections.append(
                    {
                        "image_id": image_id,
                        "category_id": dataset.cat_ids[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

            pred_mask = outputs["seg_pred"][i].cpu().numpy()
            true_mask = target["sem_mask"].numpy()
            for c in range(dataset.num_classes + 1):
                p, t = pred_mask == c, true_mask == c
                intersection[c] += np.logical_and(p, t).sum()
                union[c] += np.logical_or(p, t).sum()

            all_cls_pred.append(outputs["cls_pred"][i].cpu().numpy())
            all_cls_true.append(target["cls_labels"].numpy())

    metrics: Dict[str, float] = {}

    if coco_detections:
        coco_dt = dataset.coco.loadRes(coco_detections)
        coco_eval = COCOeval(dataset.coco, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics["detection_mAP"] = float(coco_eval.stats[0])
    else:
        metrics["detection_mAP"] = 0.0

    valid = union > 0
    metrics["seg_mIoU"] = float((intersection[valid] / union[valid]).mean()) if valid.any() else 0.0

    cls_pred = np.stack(all_cls_pred)
    cls_true = np.stack(all_cls_true)
    present = cls_true.sum(axis=0) > 0
    metrics["cls_mAP"] = float(average_precision_score(cls_true[:, present], cls_pred[:, present])) if present.any() else 0.0
    metrics["cls_F1"] = float(f1_score(cls_true, cls_pred >= 0.5, average="micro", zero_division=0))

    return metrics
