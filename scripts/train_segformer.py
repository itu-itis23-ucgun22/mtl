"""REFERANS: SegFormer'ı (HF) bizim COCO subset'inde fine-tune eder — semantic-seg specialist referansı.

Multi-task seg'imizle KIYAS için: AYNI 22.5k veri, tek görev (semantic seg), ve mIoU **bizim
engine/evaluate.py ile birebir aynı tanımda** hesaplanır (parite). ⚠️ KIYAS DEĞİL, REFERANS:
tek-görev + backbone açık (fine-tune) + farklı mimari (MiT) → "bizim veride normal modern segmenter
ne yapıyor" tavanı. RESULTS'a rejim-etiketiyle işlenir.

Girdi: CocoMultiTaskDataset (image 512 + ImageNet-norm + sem_mask). SegFormer de ImageNet-norm ister
→ uyumlu. Model: HF SegformerForSemanticSegmentation, num_labels=81 (80 sınıf + background), taze head.

    python scripts/train_segformer.py --config configs/train_ref_segformer.yaml            # MiT-B2
    python scripts/train_segformer.py --config configs/train_ref_segformer.yaml --model nvidia/mit-b0
    # kesilirse: --resume checkpoints/segformer_mit-b2_epoch7.pt

Bağımlılık: transformers (I-JEPA/BEiT'ten zaten var). ⚠️ NOT: train split'te augmentation AÇIK
(train=True; normal seg eğitimi böyle) — cache'li frozen koşularımız aug'suzdu, bu hafif bir fark
(SegFormer lehine, referans-tavan için uygun). RESULTS notunda belirtilir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mtl.config import load_config
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.datasets.collate import collate_fn
from mtl.utils.bench import measure_efficiency
from mtl.utils.device import resolve_device
from mtl.utils.results import append_result

# Not: measure_efficiency `module(x)` çağırır; HF SegFormer'da model(x) = forward(pixel_values=x)
# (pixel_values ilk positional) → SegFormer modeli doğrudan geçilebilir, sarmalayıcı gerekmez.


@torch.no_grad()
def compute_miou(model, loader, device, num_classes: int, use_amp: bool) -> float:
    """mIoU — engine/evaluate.py ile BİREBİR aynı: 0..num_classes (bg dahil), intersection/union,
    union>0 sınıflar üzerinde ortalama; pred ve true AYNI (512) çözünürlükte."""
    model.eval()
    inter = np.zeros(num_classes + 1, dtype=np.int64)
    union = np.zeros(num_classes + 1, dtype=np.int64)
    for images, targets in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(pixel_values=images).logits  # (B, 81, H/4, W/4)
        logits = F.interpolate(logits.float(), size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(1).cpu().numpy()
        for i, t in enumerate(targets):
            true = t["sem_mask"].numpy()
            pred = preds[i]
            for c in range(num_classes + 1):
                p, tt = pred == c, true == c
                inter[c] += np.logical_and(p, tt).sum()
                union[c] += np.logical_or(p, tt).sum()
    valid = union > 0
    return float((inter[valid] / union[valid]).mean()) if valid.any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="nvidia/mit-b2", help="HF SegFormer encoder (nvidia/mit-b0..b5)")
    parser.add_argument("--results-csv", default="runs/results.csv")
    parser.add_argument("--resume", default=None, help="epoch checkpoint'inden devam (veya --eval-only ile eval)")
    parser.add_argument("--eval-only", action="store_true",
                        help="EĞİTME, sadece --resume checkpoint'ini val'de eval'le (kesilmiş koşu için)")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-fps", action="store_true", help="eval'de verim (params/FPS/bellek) ölçümünü atla")
    parser.add_argument("--measure-only", action="store_true",
                        help="EĞİTME/EVAL YOK — sadece verim (params/FPS/gecikme/bellek) ölç ve çık. "
                             "Veri/checkpoint gerekmez (verim yalnız mimariye bağlı; saniyeler)")
    args = parser.parse_args()

    from transformers import SegformerForSemanticSegmentation

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    tag = args.model.split("/")[-1]  # "mit-b2"

    # --measure-only: sadece verim ölçümü — veri/eğitim/checkpoint yok (verim yalnız mimariye bağlı).
    # measure_efficiency, bizim backbone "Verimlilik" tablosuyla AYNI metodoloji (batch=1, warmup+iter, sync).
    if args.measure_only:
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model, num_labels=81, ignore_mismatched_sizes=True
        ).to(device)
        eff = measure_efficiency(model, device, img_size=cfg.data.img_size, batch=1)
        print(f"\n=== REFERANS: SegFormer ({args.model}) — verim (TAM model, batch=1, {device.type}, img {cfg.data.img_size}) ===")
        print(f"  params_M    : {eff['params_M']:.1f}")
        print(f"  latency_ms  : {eff['latency_ms']:.2f}")
        print(f"  fps         : {eff['fps']:.1f}")
        if eff['peak_mem_MB'] == eff['peak_mem_MB']:  # NaN değilse (cuda)
            print(f"  peak_mem_MB : {eff['peak_mem_MB']:.0f}")
        print("  ⚠️ A100 'Verimlilik' tablosuyla kıyaslanamaz: TAM model (yalnız backbone değil) + farklı GPU.")
        append_result(args.results_csv, {
            "run_name": f"ref_segformer_{tag}_eff", "backbone": f"segformer_{tag}",
            "trainable_layers": "ft", "checkpoint": "-", "split": "eff", "step": 0,
            "params_M": round(eff["params_M"], 2), "latency_ms": round(eff["latency_ms"], 2),
            "fps": round(eff["fps"], 1), "peak_mem_MB": round(eff["peak_mem_MB"], 0),
        })
        print(f"[results] {args.results_csv}'ye eklendi.")
        return

    train_ds = CocoMultiTaskDataset(cfg.data.ann_file, cfg.data.img_dir, img_size=cfg.data.img_size,
                                    train=True, n_images=cfg.data.n_images)
    val_ds = CocoMultiTaskDataset(cfg.data.val_ann_file, cfg.data.val_img_dir, img_size=cfg.data.img_size,
                                  train=False)
    num_classes = train_ds.num_classes  # 80
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=num_classes + 1, ignore_mismatched_sizes=True
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --eval-only: eğitmeden, verilen checkpoint'i val'de değerlendir (kesilmiş/plato yapmış koşu için)
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only için --resume <checkpoint.pt> gerekli.")
        state = torch.load(args.resume, map_location=str(device))
        model.load_state_dict(state["model"])
        step = state.get("step", 0)
        miou = compute_miou(model, val_loader, device, num_classes, use_amp)
        eff = None if args.no_fps else measure_efficiency(model, device, img_size=cfg.data.img_size, batch=1)
        print(f"\n=== REFERANS (eval-only): SegFormer ({args.model}) — bizim val (step {step}) ===")
        print(f"  seg_mIoU : {miou:.4f}   (⚠️ trained-specialist referansı; kıyas değil)")
        if eff:
            print(f"  --- verim (TAM model, batch=1, {device.type}) ---")
            print(f"  params_M    : {eff['params_M']:.1f}")
            print(f"  latency_ms  : {eff['latency_ms']:.2f}")
            print(f"  fps         : {eff['fps']:.1f}")
            if eff['peak_mem_MB'] == eff['peak_mem_MB']:
                print(f"  peak_mem_MB : {eff['peak_mem_MB']:.0f}")
            print("  ⚠️ backbone 'Verimlilik' tablosuyla kıyaslanamaz: TAM model (encoder+decode head).")
        append_result(args.results_csv, {
            "run_name": f"ref_segformer_{tag}", "backbone": f"segformer_{tag}",
            "trainable_layers": "ft", "checkpoint": args.resume, "split": "val",
            "step": step, "seg_mIoU": miou,
            **({"params_M": round(eff["params_M"], 2), "latency_ms": round(eff["latency_ms"], 2),
                "fps": round(eff["fps"], 1), "peak_mem_MB": round(eff["peak_mem_MB"], 0)} if eff else {}),
        })
        print(f"[results] {args.results_csv}'ye eklendi.")
        return

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, step = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=str(device))
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        print(f"[resume] {args.resume} → epoch {start_epoch}, step {step}")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            masks = torch.stack([t["sem_mask"] for t in targets]).to(device)  # (B,512,512), 0..80/255
            optim.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(pixel_values=images).logits
                logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, masks, ignore_index=255)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            step += 1
            if step % cfg.train.log_every == 0:
                print(f"[epoch {epoch} step {step}] seg_loss={loss.item():.4f}")
        ckpt = ckpt_dir / f"segformer_{tag}_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "step": step}, ckpt)
        print(f"[checkpoint] {ckpt}")

    miou = compute_miou(model, val_loader, device, num_classes, use_amp)
    print(f"\n=== REFERANS: SegFormer ({args.model}) bizim veride fine-tuned — bizim val ===")
    print(f"  seg_mIoU : {miou:.4f}   (⚠️ trained-specialist referansı; kıyas değil)")
    append_result(args.results_csv, {
        "run_name": f"ref_segformer_{tag}",
        "backbone": f"segformer_{tag}",
        "trainable_layers": "ft",
        "checkpoint": str(ckpt_dir / f"segformer_{tag}_epoch{cfg.train.epochs - 1}.pt"),
        "split": "val",
        "step": step,
        "seg_mIoU": miou,
    })
    print(f"[results] {args.results_csv}'ye eklendi.")


if __name__ == "__main__":
    main()
