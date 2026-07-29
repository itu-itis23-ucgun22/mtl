"""Donuk backbone'un trunk (gövde) feature'larını bir kez hesaplayıp diske yazar.

Donuk backbone (trainable_backbone_layers=0) çıktısı deterministiktir; her epoch aynıdır.
Bir kez cache'leyip sonra sadece neck+head'i cache'den eğitmek (scripts/train_cached.py),
pahalı ViT forward'ını her adımda tekrar koşmaktan kurtarır -> foundation-model sweep'i ucuzlar.

    python scripts/precompute_features.py --config configs/train_colab_dinov2.yaml --split train
    python scripts/precompute_features.py --config configs/train_colab_dinov2.yaml --split val

Cache: features/<run_name>/<split>/<idx>.npy (float16). Augmentation KAPALIDIR (flip yok),
çünkü cache tek bir görüntüye karşılık gelir; train_cached de aynı flip'siz veriyi kullanır.
Yalnızca DINO ailesi (dino_vitb16 / dinov2_*) - `trunk_forward`'ı olan backbone'lar için.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtl.config import load_config
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.datasets.collate import collate_fn
from mtl.models.backbone import build_backbone
from mtl.utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--out-dir", default=None, help="varsayılan: features/<run_name>/<split>")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.trainable_backbone_layers != 0:
        raise SystemExit(
            "Feature cache yalnızca DONUK backbone için geçerli "
            f"(trainable_backbone_layers=0). Config'te {cfg.model.trainable_backbone_layers}."
        )
    device = resolve_device(cfg.train.device)

    if args.split == "train":
        ann, img_dir = cfg.data.ann_file, cfg.data.img_dir
    else:
        ann, img_dir = cfg.data.val_ann_file, cfg.data.val_img_dir
    # train=False -> augmentation yok; cache tek bir (flip'siz) görüntüye karşılık gelir.
    # n_images: train split'i kısıtlamak için (ör. eksik indirme -> ilk N görüntüyle çalışmak).
    # val'de kısıtlama YOK (değerlendirme tam split üzerinde kalmalı).
    dataset = CocoMultiTaskDataset(
        ann, img_dir, img_size=cfg.data.img_size, train=False,
        n_images=cfg.data.n_images if args.split == "train" else None,
    )

    backbone = build_backbone(
        cfg.model.backbone_name, pretrained=cfg.model.pretrained, trainable_layers=0,
        multilayer_taps=cfg.model.multilayer_taps,
    ).to(device)
    backbone.eval()
    if not hasattr(backbone, "trunk_forward"):
        raise SystemExit(
            f"'{cfg.model.backbone_name}' backbone'unda trunk_forward yok - feature cache "
            "yalnızca DINO ailesi (dino_vitb16 / dinov2_*) için (ResNet donuk gövdesi zaten ucuz)."
        )

    out_dir = Path(args.out_dir or f"features/{cfg.train.run_name}/{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or cfg.train.batch_size
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )

    idx = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"precompute {args.split}"):
            trunk = backbone.trunk_forward(images.to(device))  # (B, embed, h, w)
            trunk = trunk.to(torch.float16).cpu().numpy()
            for j in range(trunk.shape[0]):
                np.save(out_dir / f"{idx}.npy", trunk[j])
                idx += 1

    sample = np.load(out_dir / "0.npy")
    per_mb = sample.nbytes / 1e6
    print(f"\n{idx} görsel cache'lendi -> {out_dir}")
    print(f"Her biri {sample.shape} float16 ~{per_mb:.1f} MB; toplam ~{per_mb * idx / 1000:.1f} GB")
    print("Sonraki: python scripts/train_cached.py --config <config>")


if __name__ == "__main__":
    main()
