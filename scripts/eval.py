"""CLI: load a checkpoint and run engine.evaluate on the val split.

    python scripts/eval.py --config configs/train_colab_gpu.yaml --checkpoint checkpoints/colab_gpu_epoch15.pt

Metrikleri hizalı bir tablo olarak basar ve runs/results.csv'ye bir satır ekler
(hangi omurga/checkpoint/adım + dört metrik) - böylece ResNet ve DINO koşuları tek
dosyada birikir ve scripts/compare_results.py ile karşılaştırılabilir.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mtl.config import load_config
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.datasets.collate import collate_fn
from mtl.engine.checkpoint import load_checkpoint
from mtl.engine.evaluate import evaluate
from mtl.models.multitask_model import MultiTaskModel
from mtl.utils.device import resolve_device
from mtl.utils.results import append_result

METRIC_KEYS = ["detection_mAP", "seg_mIoU", "cls_mAP", "cls_F1"]


def print_metrics_table(metrics: dict, backbone: str, checkpoint: str) -> None:
    print(f"\n=== Eval: backbone={backbone}  checkpoint={checkpoint} ===")
    width = max(len(k) for k in metrics)
    for key in METRIC_KEYS:
        if key in metrics:
            print(f"  {key:<{width}} : {metrics[key]:.4f}")
    # METRIC_KEYS dışında bir metrik eklenirse yine de görünsün
    for key, value in metrics.items():
        if key not in METRIC_KEYS:
            print(f"  {key:<{width}} : {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--results-csv", default="runs/results.csv",
        help="toplu karşılaştırma CSV'si (varsayılan runs/results.csv)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)

    dataset = CocoMultiTaskDataset(
        ann_file=cfg.data.val_ann_file,
        img_dir=cfg.data.val_img_dir,
        img_size=cfg.data.img_size,
        train=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )

    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name,
        pretrained=False,
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        det_num_classes=dataset.num_classes,
        seg_num_classes=dataset.num_classes + 1,
        cls_num_labels=dataset.num_classes,
    ).to(device)
    step = load_checkpoint(model, optimizer=None, path=args.checkpoint, map_location=str(device))

    metrics = evaluate(model, dataset, dataloader, device)

    print_metrics_table(metrics, cfg.model.backbone_name, args.checkpoint)
    append_result(
        args.results_csv,
        {
            "run_name": cfg.train.run_name,
            "backbone": cfg.model.backbone_name,
            "trainable_layers": cfg.model.trainable_backbone_layers,
            "checkpoint": args.checkpoint,
            "step": step,  # checkpoint'e kaydedilen adım/epoch (checkpoint.py)
            **metrics,
        },
    )
    print(f"\n[results] {args.results_csv}'ye eklendi.")


if __name__ == "__main__":
    main()
