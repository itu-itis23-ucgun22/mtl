"""CLI: load a checkpoint and run engine.evaluate on the val split.

    python scripts/eval.py --config configs/train_colab_gpu.yaml --checkpoint checkpoints/colab_gpu_epoch15.pt
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
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
    load_checkpoint(model, optimizer=None, path=args.checkpoint, map_location=str(device))

    metrics = evaluate(model, dataset, dataloader, device)
    print(metrics)


if __name__ == "__main__":
    main()
