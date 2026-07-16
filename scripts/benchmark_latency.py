"""Her omurganın DONUK çıkarım maliyetini ölçer: parametre · gecikme (ms) · verim · tepe bellek.

Motivasyon (kısıtlı platform / uçak): omurga en pahalı parça. Bu script paylaşılabilir
omurga adaylarını *aynı* protokolde (batch=1, native img_size, ısınma + ortalama) kıyaslar;
"hangi ön-eğitim daha iyi transfer eder" tablosunun yanına "hangisi ne kadar pahalı" sütununu koyar.

Ölçülen: backbone+neck ileri-geçişi (build_backbone çıktısı) — yani üç görevin PAYLAŞTIĞI kısım.
Görev başlıkları hafif ve hepsinde ortak olduğundan omurga maliyeti belirleyicidir.

Her config için bir kez koş (eval/precompute gibi); runs/latency.csv'ye satır ekler:

    python scripts/benchmark_latency.py --config configs/train_colab_dinov2.yaml
    python scripts/benchmark_latency.py --config configs/train_colab_mae.yaml
    ... (her omurga için)

Not: --pretrained vermezsen ağırlıklar İNDİRİLMEZ (rastgele init) — gecikme yalnızca MİMARİYE
bağlı olduğundan bu ölçümü değiştirmez ve HF indirmesini atlar. Sadece I-JEPA/BEiT'te doğru
mimari için küçük bir config.json çekilebilir (MTL_WEIGHTS_DIR ayarlıysa o da yerelden okunur).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from mtl.config import load_config
from mtl.models.backbone import build_backbone
from mtl.utils.device import resolve_device


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def _time_forward(model, x, iters: int, is_cuda: bool) -> float:
    """iters kez ileri-geçiş; toplam saniye döndürür (senkronizasyon dahil)."""
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch", type=int, default=1,
                        help="çıkarım batch'i (edge/uçak için 1 gerçekçi; default 1)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true",
                        help="gerçek ağırlıkları indir (gecikmeyi değiştirmez; genelde gereksiz)")
    parser.add_argument("--out-csv", default="runs/latency.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    is_cuda = device.type == "cuda"
    name = cfg.model.backbone_name
    img = cfg.data.img_size

    backbone = build_backbone(name, pretrained=args.pretrained, trainable_layers=0).to(device)
    backbone.eval()
    params_m = _count_params(backbone) / 1e6

    x = torch.randn(args.batch, 3, img, img, device=device)

    # ısınma (cuDNN autotune, lazy init, ilk-çağrı derlemeleri)
    _time_forward(backbone, x, args.warmup, is_cuda)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    total_s = _time_forward(backbone, x, args.iters, is_cuda)

    per_iter_ms = total_s / args.iters * 1000.0
    per_img_ms = per_iter_ms / args.batch
    img_per_s = args.batch * args.iters / total_s
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if is_cuda else float("nan")
    gpu = torch.cuda.get_device_name(device) if is_cuda else "cpu"

    print(f"\n=== {name}  (img {img}, batch {args.batch}) ===")
    print(f"  parametre     : {params_m:8.1f} M")
    print(f"  gecikme/görsel: {per_img_ms:8.2f} ms")
    print(f"  batch gecikme : {per_iter_ms:8.2f} ms")
    print(f"  verim         : {img_per_s:8.1f} görsel/s")
    if is_cuda:
        print(f"  tepe bellek   : {peak_mb:8.0f} MB   ({gpu})")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["backbone", "img_size", "batch", "params_M",
                        "latency_ms_per_img", "latency_ms_per_batch",
                        "throughput_img_s", "peak_mem_MB", "device"])
        w.writerow([name, img, args.batch, f"{params_m:.2f}",
                    f"{per_img_ms:.2f}", f"{per_iter_ms:.2f}",
                    f"{img_per_s:.1f}", f"{peak_mb:.0f}" if is_cuda else "", gpu])
    print(f"  -> {out} güncellendi")


if __name__ == "__main__":
    main()
