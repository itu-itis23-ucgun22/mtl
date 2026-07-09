"""Cache'lenmiş DONUK backbone feature'larından eğitim: neck + head'leri eğitir, backbone'u atlar.

    # 1) trunk feature'larını bir kez yaz (donuk backbone):
    python scripts/precompute_features.py --config configs/train_colab_dinov2.yaml --split train
    # 2) neck+head'i cache'den eğit (pahalı ViT forward YOK):
    python scripts/train_cached.py --config configs/train_colab_dinov2.yaml
    # 3) normal eval (görüntü tabanlı forward, donuk backbone + eğitilmiş neck/head):
    python scripts/eval.py --config configs/train_colab_dinov2.yaml --checkpoint checkpoints/<run>_cached_epoch15.pt

Not: Augmentation yoktur (cache flip'siz). Checkpoint tam model state'ini içerir (donuk ViT
ağırlıkları da) -> normal eval.py ile uyumludur. ViT-B ~344 MB/checkpoint olduğu için burada
yalnızca EPOCH sonu checkpoint'i yazılır (cached eğitim hızlı; ara checkpoint gerekmez).
Oturum koparsa --resume checkpoints/<run>_cached_epoch<N>.pt ile kaldığın epoch'tan devam:
tamamlanan epoch'lar atlanır (ör. epoch12 -> epoch13'ten sürer).
Foundation-model sweep'inde her donuk backbone için bir kez precompute + hızlı head eğitimi:
bkz. ROADMAP.md.
"""
from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from mtl.config import config_to_dict, load_config
from mtl.datasets.cached_features import CachedFeatureDataset
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.datasets.collate import collate_fn
from mtl.engine.checkpoint import load_checkpoint, save_checkpoint
from mtl.losses.joint_loss import combine_losses
from mtl.models.multitask_model import MultiTaskModel
from mtl.utils.device import resolve_device
from mtl.utils.logging import CsvLogger
from mtl.utils.seed import set_seed


def _move_targets(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features-dir", default=None, help="varsayılan: features/<run_name>/train")
    parser.add_argument(
        "--resume", default=None,
        help="epoch checkpoint'inden devam et (ör. checkpoints/colab_clip_cached_epoch12.pt); "
             "tamamlanan epoch'lar atlanır, kaldığın epoch'tan devam edilir.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.trainable_backbone_layers != 0:
        raise SystemExit(
            "train_cached yalnızca DONUK backbone için (trainable_backbone_layers=0); "
            "çözük backbone'da cache geçersizdir, normal scripts/train.py kullan."
        )
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    base = CocoMultiTaskDataset(
        cfg.data.ann_file, cfg.data.img_dir, img_size=cfg.data.img_size, train=False
    )
    feat_dir = args.features_dir or f"features/{cfg.train.run_name}/train"
    dataset = CachedFeatureDataset(feat_dir, base)
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=cfg.data.num_workers > 0,
    )

    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
        trainable_backbone_layers=0,
        det_num_classes=base.num_classes,
        seg_num_classes=base.num_classes + 1,
        cls_num_labels=base.num_classes,
    ).to(device)

    # Yalnızca gradyanı olan parametreler (neck + head'ler); donuk ViT güncellenmez.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    logger = CsvLogger(out_dir="runs", run_name=cfg.train.run_name + "_cached")

    # --resume: kaydedilen global adımdan devam. train_cached checkpoint'i epoch SONUNDA
    # step=(epoch+1)*steps_per_epoch ile yazar; bu yüzden step//steps_per_epoch tam olarak
    # bir sonraki (yarım kalmamış) epoch'u verir. Tamamlanan epoch'lar atlanır.
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, args.resume, map_location=str(device))
        print(f"Resumed from {args.resume} @ global step {start_step}")
    steps_per_epoch = math.ceil(len(dataset) / cfg.train.batch_size)
    start_epoch = start_step // steps_per_epoch

    print("Config:", config_to_dict(cfg))
    print(f"Feature cache: {feat_dir} | eğitilebilir tensör sayısı: {len(params)}")

    img_hw = (cfg.data.img_size, cfg.data.img_size)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    model.train()
    step = start_step
    for epoch in range(start_epoch, cfg.train.epochs):
        for trunk, targets in loader:
            trunk = trunk.to(device)
            targets = _move_targets(targets, device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
                loss_dict = model.forward_from_trunk(trunk, img_hw, targets)
                total_loss, raw = combine_losses(loss_dict, cfg.loss)

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {raw}")

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.train.log_every == 0:
                logger.log(step, raw)
            step += 1

        save_checkpoint(
            model, optimizer, epoch,
            f"{cfg.train.checkpoint_dir}/{cfg.train.run_name}_cached_epoch{epoch}.pt",
            step=step,
        )
        print(f"[epoch {epoch}] bitti, step {step}")

    print(f"Training finished after {step} steps.")


if __name__ == "__main__":
    main()
