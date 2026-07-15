"""Kalitatif inference görselleştirme: model GERÇEKTEN kutu çizebiliyor / segmentleyebiliyor mu?

Metrikler "ne kadar iyi" der; bu script "GÖSTER" der. Bir veya birden çok backbone'u
(config+checkpoint) AYNI val görüntülerinde koşturup her görüntü için tek bir figür üretir:
  - satır 0: GROUND TRUTH (gerçek kutular + gerçek segmentasyon maskesi)
  - her backbone için bir satır: tahmin edilen kutular | tahmin edilen maske
  - başlıklarda tahmin edilen sınıf etiketleri (multi-label cls) + skorlar
Rapora koymak için PNG kaydeder (--out-dir Drive olabilir -> kalıcı).

    # tek backbone, 6 görüntü:
    python scripts/visualize.py --config configs/train_colab_dinov2.yaml \
        --checkpoint checkpoints/colab_dinov2_epoch15.pt --num-images 6 --out-dir viz/dinov2

    # ÇOK backbone karşılaştırması (aynı görüntüler, yan yana) - rapor için en etkilisi:
    python scripts/visualize.py \
        --config  configs/train_colab_dinov2.yaml configs/train_colab_sam.yaml configs/train_colab_mae.yaml \
        --checkpoint checkpoints/colab_dinov2_epoch15.pt checkpoints/colab_sam_cached_epoch15.pt \
                     checkpoints/colab_mae_cached_epoch15.pt \
        --num-images 6 --out-dir viz/compare

Not: model pretrained=False ile kurulur (ağırlıklar checkpoint'ten gelir) -> backbone ağırlığı
HF'ten İNMEZ, indirme derdi olmaz. Config'lerin hepsi aynı val setine baktığı için --num-images
ilk N görüntü tüm backbone'larda AYNIdır (yan yana kıyas doğru).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # başsız (Colab/script) kayıt için
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtl.config import load_config
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.engine.checkpoint import load_checkpoint
from mtl.models.multitask_model import MultiTaskModel
from mtl.utils.device import resolve_device

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(image: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-norm tensör -> (H,W,3) [0,1] görüntü (gösterim için)."""
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def class_colors(n: int) -> np.ndarray:
    """Sınıf başına sabit renk (tüm backbone/görüntülerde tutarlı olsun diye seed'li)."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.2, 1.0, size=(n + 1, 3))  # +1: index 0 = arka plan


def seg_overlay(img: np.ndarray, seg: np.ndarray, colors: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """seg (H,W) sınıf indeksleri (0=bg) -> renkli maskeyi görüntü üstüne bindirir."""
    out = img.copy()
    for cls in np.unique(seg):
        if cls == 0:
            continue  # arka plan çizilmez
        m = seg == cls
        out[m] = (1 - alpha) * out[m] + alpha * colors[cls]
    return np.clip(out, 0, 1)


def draw_boxes(ax, boxes, labels, scores, cat_names, colors, thresh):
    n = 0
    for i in range(len(boxes)):
        if scores is not None and scores[i] < thresh:
            continue
        x1, y1, x2, y2 = boxes[i].tolist()
        cls = int(labels[i])
        color = colors[cls + 1]  # cls 0..K-1 -> renk indexi cls+1 (seg ile aynı palet)
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        tag = cat_names[cls] + (f" {scores[i]:.2f}" if scores is not None else "")
        ax.text(x1, max(y1 - 2, 0), tag, color="white", fontsize=7,
                bbox=dict(facecolor=color, edgecolor="none", pad=0.5, alpha=0.8))
        n += 1
    return n


def cls_text(cls_pred, cat_names, topk=5):
    probs, idx = torch.topk(cls_pred, min(topk, len(cls_pred)))
    return ", ".join(f"{cat_names[int(i)]}({p:.2f})" for p, i in zip(probs.tolist(), idx.tolist()))


def build_model(cfg, checkpoint, dataset, device):
    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name,
        pretrained=False,  # ★ ağırlıklar checkpoint'ten -> HF'ten indirme YOK
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        det_num_classes=dataset.num_classes,
        seg_num_classes=dataset.num_classes + 1,
        cls_num_labels=dataset.num_classes,
    ).to(device)
    load_checkpoint(model, optimizer=None, path=checkpoint, map_location=str(device))
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True, help="bir veya çok config (backbone başına)")
    p.add_argument("--checkpoint", nargs="+", required=True, help="config'lerle AYNI sırada checkpoint")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--indices", type=int, nargs="+", default=None, help="belirli indeksler (yoksa ilk N)")
    p.add_argument("--score-thresh", type=float, default=0.3, help="kutu çizim eşiği")
    p.add_argument("--ann-file", default=None, help="split JSON (yoksa config'in val'i). TEST için test_subset.json")
    p.add_argument("--img-dir", default=None, help="--ann-file ile eşleşen görüntü klasörü")
    p.add_argument("--out-dir", default="viz")
    args = p.parse_args()

    if len(args.config) != len(args.checkpoint):
        raise SystemExit("--config ve --checkpoint aynı sayıda olmalı (backbone başına bir çift)")

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Her backbone için: config, val dataset (kendi img_size'ında), model.
    entries = []
    for cfg_path, ckpt in zip(args.config, args.checkpoint):
        cfg = load_config(cfg_path)
        ann = args.ann_file or cfg.data.val_ann_file  # test için --ann-file ver
        img_dir = args.img_dir or cfg.data.val_img_dir
        ds = CocoMultiTaskDataset(ann, img_dir, img_size=cfg.data.img_size, train=False)
        model = build_model(cfg, ckpt, ds, device)
        entries.append((cfg.model.backbone_name, ds, model))
        print(f"yüklendi: {cfg.model.backbone_name}  <- {ckpt}")

    # Kategori isimleri (ilk dataset'ten; hepsi aynı val ann -> aynı sınıflar).
    ref_ds = entries[0][1]
    cats = ref_ds.coco.loadCats(ref_ds.cat_ids)
    cat_names = [c["name"] for c in cats]
    colors = class_colors(ref_ds.num_classes)

    indices = args.indices if args.indices is not None else list(range(args.num_images))

    for idx in indices:
        n_rows = 1 + len(entries)  # GT + her backbone
        fig, axes = plt.subplots(n_rows, 2, figsize=(9, 4.2 * n_rows))
        axes = np.atleast_2d(axes)

        # --- satır 0: GROUND TRUTH ---
        img0, tgt0 = ref_ds[idx]
        img0_np = denormalize(img0)
        image_id = int(tgt0["image_id"].item())

        axes[0, 0].imshow(img0_np)
        ng = draw_boxes(axes[0, 0], tgt0["boxes"], tgt0["labels"], None, cat_names, colors, 0)
        axes[0, 0].set_title(f"GT detection ({ng} nesne)  |  img_id={image_id}", fontsize=9)
        axes[0, 1].imshow(seg_overlay(img0_np, tgt0["sem_mask"].cpu().numpy(), colors))
        gt_cls = [cat_names[i] for i in torch.nonzero(tgt0["cls_labels"]).flatten().tolist()]
        axes[0, 1].set_title("GT segmentation\nGT sınıflar: " + ", ".join(gt_cls[:8]), fontsize=8)

        # --- her backbone için bir satır ---
        for r, (name, ds, model) in enumerate(entries, start=1):
            img, _ = ds[idx]
            img_np = denormalize(img)
            with torch.no_grad():
                out = model(img.unsqueeze(0).to(device))
            det = out["detections"][0]
            seg = out["seg_pred"][0].cpu().numpy()
            cls = out["cls_pred"][0].cpu()

            axes[r, 0].imshow(img_np)
            nd = draw_boxes(axes[r, 0], det["boxes"].cpu(), det["labels"].cpu(),
                            det["scores"].cpu(), cat_names, colors, args.score_thresh)
            axes[r, 0].set_title(f"{name} — detection ({nd} kutu > {args.score_thresh})", fontsize=9)
            axes[r, 1].imshow(seg_overlay(img_np, seg, colors))
            axes[r, 1].set_title(f"{name} — segmentation\ncls: " + cls_text(cls, cat_names), fontsize=8)

        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        dst = out_dir / f"img{idx:03d}_id{image_id}.png"
        fig.savefig(dst, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"kaydedildi: {dst}")

    print(f"\nBitti -> {out_dir} ({len(indices)} görüntü, {len(entries)} backbone).")


if __name__ == "__main__":
    main()
