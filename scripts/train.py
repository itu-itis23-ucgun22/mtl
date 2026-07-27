"""CLI entry point for both local CPU smoke runs and Colab/Kaggle GPU runs -
the same script, driven entirely by --config and --overrides.

    python scripts/train.py --config configs/smoke_cpu.yaml
    python scripts/train.py --config configs/train_colab_gpu.yaml --overrides train.device=cuda train.batch_size=16
    python scripts/train.py --config configs/train_colab_gpu.yaml --smoke   # sanity-check the GPU config on CPU
"""
from __future__ import annotations

import argparse
import math

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
        pin_memory=True,  # CPU->GPU kopyayı hızlandırır (GPU'yu veri beklerken boşta bırakmamak için)
        persistent_workers=cfg.data.num_workers > 0,  # her epoch başında worker'ları yeniden kurma
    )

    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
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
        adaptive_loss=cfg.loss.adaptive,
    ).to(device)

    # LoRA/donuk backbone'da taban parametreler requires_grad=False → optimizer yalnız
    # eğitilebilir (LoRA + neck + head) tensörleri alsın (frozen olanlar zaten grad almaz).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    logger = CsvLogger(out_dir="runs", run_name=cfg.train.run_name)

    start_step = 0
    if args.resume:
        # Warm-starts weights+optimizer from a checkpoint (saved either mid-epoch
        # via checkpoint_every_steps, or at a previous epoch's end) and recovers
        # the global step so training continues across sessions/accounts. The
        # partially-done epoch is re-iterated from the dataloader start rather
        # than resuming an exact dataloader position - simpler, and fine since
        # the model/optimizer state is what actually matters.
        start_step = load_checkpoint(model, optimizer, args.resume, map_location=str(device))
        print(f"Resumed from {args.resume} at global step {start_step}")

    print("Config:", config_to_dict(cfg))

    # Global adım + tamamlanan epoch'ları geri yükle: --resume kaldığı yerden devam eder
    # (oturumlar/hesaplar arası). Tamamlanan epoch'lar atlanır; kısmen biten epoch tam olarak
    # yeniden koşulur (dataloader baştan gezilir - checkpoint.py notu), adım sayacı süreklidir.
    steps_per_epoch = math.ceil(len(dataset) / cfg.train.batch_size)
    start_epoch = start_step // steps_per_epoch if steps_per_epoch else 0

    step = start_step
    for epoch in range(start_epoch, cfg.train.epochs):
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
            model, optimizer, epoch,
            f"{cfg.train.checkpoint_dir}/{cfg.train.run_name}_epoch{epoch}.pt",
            step=step,
        )

    print(f"Training finished after {step} steps.")


if __name__ == "__main__":
    main()
