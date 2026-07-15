"""Epoch-başı METRİK eğrileri: kaydedilmiş epoch checkpoint'lerini eval'leyip metrik-vs-epoch çizer.

Bu bir ÖĞRENME EĞRİSİ (loss değil, DOĞRULUK metrikleri: det_mAP / seg_mIoU / cls_mAP / cls_F1).
Adım-adım eğitim loss'u `runs/*.csv`'de (ephemeral); bu ise checkpoint'lerden GERİYE DÖNÜK çıkarılır
-> runtime resetlense bile checkpoint'ler Drive'da olduğu sürece eğri kurtarılabilir. Rapor için:
her görevin epoch boyunca nasıl geliştiğini (ve platoya oturup oturmadığını) gösterir.

    # tüm epoch checkpoint'lerini eval'le + çiz (glob TIRNAK içinde):
    python scripts/plot_metric_curves.py --config configs/train_colab_mae.yaml \
        --checkpoints "checkpoints/colab_mae_cached_epoch*.pt" \
        --out-dir /content/drive/MyDrive/mtl_data/viz/curves

    # hız için val'i altörnekle (eğilim yine görünür):
    python scripts/plot_metric_curves.py --config configs/train_colab_dinov2.yaml \
        --checkpoints "checkpoints/colab_dinov2_epoch*.pt" --max-images 500 --out-dir viz/curves

Not: her epoch için bir eval koşar (2000 val görüntüsü -> ~30-60 sn/epoch). 16 epoch = ~10-15 dk;
--max-images ile hızlandırılabilir. Metrikler ayrıca <out-dir>/<run>_curve.csv'ye yazılır (tekrar
eval etmeden yeniden çizebilmen için).
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from mtl.config import load_config
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.datasets.collate import collate_fn
from mtl.engine.checkpoint import load_checkpoint
from mtl.engine.evaluate import evaluate
from mtl.models.multitask_model import MultiTaskModel
from mtl.utils.device import resolve_device

METRIC_KEYS = ["detection_mAP", "seg_mIoU", "cls_mAP", "cls_F1"]


def epoch_of(path: str) -> int:
    """Dosya adından epoch numarasını çıkar (..._epoch12.pt -> 12); bulunamazsa büyük sayı."""
    m = re.search(r"epoch(\d+)", Path(path).name)
    return int(m.group(1)) if m else 10**9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="epoch checkpoint'leri: glob (\"...epoch*.pt\", tırnak içinde) ya da açık liste")
    p.add_argument("--max-images", type=int, default=None, help="hız için val altörnekleme (ör. 500)")
    p.add_argument("--out-dir", default="viz/curves")
    args = p.parse_args()

    # glob'ları genişlet (shell genişletmediyse), epoch'a göre sırala
    paths = []
    for c in args.checkpoints:
        paths.extend(glob.glob(c) if any(ch in c for ch in "*?[") else [c])
    paths = sorted(set(paths), key=epoch_of)
    if not paths:
        raise SystemExit(f"checkpoint bulunamadı: {args.checkpoints}")
    print("eval edilecek checkpoint'ler:", *[Path(p).name for p in paths], sep="\n  ")

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)

    dataset = CocoMultiTaskDataset(cfg.data.val_ann_file, cfg.data.val_img_dir,
                                   img_size=cfg.data.img_size, train=False)
    if args.max_images:
        dataset.img_ids = dataset.img_ids[:args.max_images]
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=collate_fn)

    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name, pretrained=False,
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        det_num_classes=dataset.num_classes, seg_num_classes=dataset.num_classes + 1,
        cls_num_labels=dataset.num_classes,
    ).to(device)

    rows = []
    for path in paths:
        load_checkpoint(model, optimizer=None, path=path, map_location=str(device))
        model.eval()
        metrics = evaluate(model, dataset, loader, device)
        ep = epoch_of(path)
        rows.append({"epoch": ep, **{k: metrics.get(k) for k in METRIC_KEYS}})
        print(f"[epoch {ep}] " + "  ".join(f"{k}={metrics.get(k):.4f}" for k in METRIC_KEYS))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run = cfg.train.run_name

    # CSV: tekrar eval etmeden yeniden çizmek için
    csv_path = out_dir / f"{run}_curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + METRIC_KEYS)
        w.writeheader(); w.writerows(rows)

    # Çizim: dört metrik tek figürde (hepsi 0..~0.8 aralığında, kıyas kolay)
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in METRIC_KEYS:
        ax.plot(epochs, [r[key] for r in rows], marker="o", label=key)
    ax.set_xlabel("epoch"); ax.set_ylabel("metrik"); ax.set_ylim(0, None)
    ax.set_title(f"{cfg.model.backbone_name} — epoch-başı metrik eğrisi (val)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    png = out_dir / f"{run}_curve.png"
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)

    print(f"\nkaydedildi:\n  {png}\n  {csv_path}")


if __name__ == "__main__":
    main()
