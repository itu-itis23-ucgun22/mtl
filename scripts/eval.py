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
from mtl.utils.bench import measure_efficiency
from mtl.utils.device import resolve_device
from mtl.utils.results import append_result

METRIC_KEYS = ["detection_mAP", "seg_mIoU", "cls_mAP", "cls_F1"]


def print_metrics_table(metrics: dict, backbone: str, checkpoint: str, eff: dict | None = None) -> None:
    print(f"\n=== Eval: backbone={backbone}  checkpoint={checkpoint} ===")
    width = max(len(k) for k in metrics)
    for key in METRIC_KEYS:
        if key in metrics:
            print(f"  {key:<{width}} : {metrics[key]:.4f}")
    # METRIC_KEYS dışında bir metrik eklenirse yine de görünsün
    for key, value in metrics.items():
        if key not in METRIC_KEYS:
            print(f"  {key:<{width}} : {value:.4f}")
    if eff:
        print(f"  --- verim (backbone, batch=1) ---")
        print(f"  {'params_M':<{width}} : {eff['params_M']:.1f}")
        print(f"  {'latency_ms':<{width}} : {eff['latency_ms']:.2f}")
        print(f"  {'fps':<{width}} : {eff['fps']:.1f}")
        if eff['peak_mem_MB'] == eff['peak_mem_MB']:  # NaN değilse (cuda)
            print(f"  {'peak_mem_MB':<{width}} : {eff['peak_mem_MB']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--results-csv", default="runs/results.csv",
        help="toplu karşılaştırma CSV'si (varsayılan runs/results.csv)",
    )
    parser.add_argument(
        "--ann-file", default=None,
        help="değerlendirilecek split'in JSON'ı; verilmezse config'in val'i. TEST için: "
             "--ann-file data/coco_subset/annotations/instances_test_subset.json",
    )
    parser.add_argument("--img-dir", default=None, help="--ann-file ile eşleşen görüntü klasörü (test için)")
    parser.add_argument("--split-name", default=None, help="results.csv'ye yazılacak etiket (ör. 'test')")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="verim ölçümünü (params/gecikme/fps) atla")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)

    ann_file = args.ann_file or cfg.data.val_ann_file
    img_dir = args.img_dir or cfg.data.val_img_dir
    dataset = CocoMultiTaskDataset(
        ann_file=ann_file,
        img_dir=img_dir,
        img_size=cfg.data.img_size,
        train=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
        pin_memory=True,  # CPU->GPU kopyayı hızlandırır
        persistent_workers=cfg.data.num_workers > 0,
    )

    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name,
        pretrained=False,
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        det_num_classes=dataset.num_classes,
        seg_num_classes=dataset.num_classes + 1,
        cls_num_labels=dataset.num_classes,
        lora=cfg.model.lora,
        lora_rank=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_targets=cfg.model.lora_targets,
        lora_blocks=cfg.model.lora_blocks,
    ).to(device)
    step = load_checkpoint(model, optimizer=None, path=args.checkpoint, map_location=str(device))

    metrics = evaluate(model, dataset, dataloader, device)

    # Verim (backbone çıkarım maliyeti) — metriklerle AYNI çıktıda; kalite + maliyet birlikte.
    eff = None
    if not args.no_benchmark:
        eff = measure_efficiency(model.backbone, device, img_size=cfg.data.img_size, batch=1)

    print_metrics_table(metrics, cfg.model.backbone_name, args.checkpoint, eff)
    append_result(
        args.results_csv,
        {
            "run_name": cfg.train.run_name + (f"_{args.split_name}" if args.split_name else ""),
            "backbone": cfg.model.backbone_name,
            "trainable_layers": cfg.model.trainable_backbone_layers,
            "checkpoint": args.checkpoint,
            "split": args.split_name or ("test" if args.ann_file else "val"),
            "step": step,  # checkpoint'e kaydedilen adım/epoch (checkpoint.py)
            **metrics,
            **({"params_M": round(eff["params_M"], 2), "latency_ms": round(eff["latency_ms"], 2),
                "fps": round(eff["fps"], 1), "peak_mem_MB": round(eff["peak_mem_MB"], 0)} if eff else {}),
        },
    )
    print(f"\n[results] {args.results_csv}'ye eklendi.")


if __name__ == "__main__":
    main()
