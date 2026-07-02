"""CLI entry point for both local CPU smoke runs and Colab/Kaggle GPU runs -
the same script, driven entirely by --config and --overrides.

    python scripts/train.py --config configs/smoke_cpu.yaml
    python scripts/train.py --config configs/train_colab_gpu.yaml --overrides train.device=cuda train.batch_size=16
    python scripts/train.py --config configs/train_colab_gpu.yaml --smoke   # sanity-check the GPU config on CPU
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mtl.config import apply_smoke_overrides, config_to_dict, load_config
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.datasets.collate import collate_fn
from mtl.engine.checkpoint import load_checkpoint, save_checkpoint
from mtl.engine.train_one_epoch import train_one_epoch
from mtl.models.multitask_model import MultiTaskModel
from mtl.utils.device import resolve_device
from mtl.utils.logging import CsvLogger
from mtl.utils.seed import set_seed


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", nargs="*", default=[], help="section.field=value pairs")
    parser.add_argument("--smoke", action="store_true", help="force a tiny, fast, CPU-safe run")
    parser.add_argument("--resume", help="path to a checkpoint to warm-start model+optimizer weights from")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    dataset = CocoMultiTaskDataset(
        ann_file=cfg.data.ann_file,
        img_dir=cfg.data.img_dir,
        img_size=cfg.data.img_size,
        train=True,
        n_images=cfg.data.n_images,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        det_num_classes=dataset.num_classes,
        seg_num_classes=dataset.num_classes + 1,
        cls_num_labels=dataset.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    logger = CsvLogger(out_dir="runs", run_name=cfg.train.run_name)

    if args.resume:
        # Warm-starts weights from a checkpoint (saved either mid-epoch via
        # checkpoint_every_steps, or at a previous epoch's end). This
        # re-iterates the dataloader from the start rather than resuming an
        # exact dataloader position - simpler, and fine for interruption
        # recovery since the model/optimizer state is what actually matters.
        load_checkpoint(model, optimizer, args.resume, map_location=str(device))
        print(f"Resumed weights from {args.resume}")

    print("Config:", config_to_dict(cfg))

    step = 0
    for epoch in range(cfg.train.epochs):
        remaining_steps = None
        if cfg.train.max_steps is not None:
            remaining_steps = cfg.train.max_steps - step
            if remaining_steps <= 0:
                break
        step = train_one_epoch(
            model,
            dataloader,
            optimizer,
            cfg.loss,
            device,
            logger,
            max_steps=remaining_steps,
            log_every=cfg.train.log_every,
            amp=cfg.train.amp,
            start_step=step,
            checkpoint_every_steps=cfg.train.checkpoint_every_steps,
            checkpoint_dir=cfg.train.checkpoint_dir,
            run_name=cfg.train.run_name,
        )
        save_checkpoint(
            model, optimizer, epoch, f"{cfg.train.checkpoint_dir}/{cfg.train.run_name}_epoch{epoch}.pt"
        )

    print(f"Training finished after {step} steps.")


if __name__ == "__main__":
    main()
