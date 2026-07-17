"""Her omurganın DONUK çıkarım maliyetini ölçer: parametre · gecikme (ms) · FPS · tepe bellek.

Motivasyon (kısıtlı platform / uçak): omurga en pahalı parça. Bu script paylaşılabilir
omurga adaylarını *aynı* protokolde (batch=1, native img_size, ısınma + ortalama) kıyaslar;
"hangi ön-eğitim daha iyi transfer eder" tablosunun yanına "hangisi ne kadar pahalı" sütununu koyar.

Ölçülen: backbone+neck ileri-geçişi (build_backbone çıktısı) — yani üç görevin PAYLAŞTIĞI kısım.
Ölçüm mantığı mtl.utils.bench'te (eval.py da AYNI fonksiyonu kullanır → iki yerde aynı sayı).

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
from pathlib import Path

import torch

from mtl.config import load_config
from mtl.models.backbone import build_backbone
from mtl.utils.bench import measure_efficiency
from mtl.utils.device import resolve_device


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
    name = cfg.model.backbone_name
    img = cfg.data.img_size

    backbone = build_backbone(name, pretrained=args.pretrained, trainable_layers=0).to(device)
    eff = measure_efficiency(backbone, device, img_size=img, batch=args.batch,
                             warmup=args.warmup, iters=args.iters)
    is_cuda = device.type == "cuda"
    gpu = torch.cuda.get_device_name(device) if is_cuda else "cpu"

    print(f"\n=== {name}  (img {img}, batch {args.batch}) ===")
    print(f"  parametre     : {eff['params_M']:8.1f} M")
    print(f"  gecikme/görsel: {eff['latency_ms']:8.2f} ms")
    print(f"  FPS           : {eff['fps']:8.1f} kare/s")
    if is_cuda:
        print(f"  tepe bellek   : {eff['peak_mem_MB']:8.0f} MB   ({gpu})")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["backbone", "img_size", "batch", "params_M",
                        "latency_ms_per_img", "fps", "peak_mem_MB", "device"])
        w.writerow([name, img, args.batch, f"{eff['params_M']:.2f}",
                    f"{eff['latency_ms']:.2f}", f"{eff['fps']:.1f}",
                    f"{eff['peak_mem_MB']:.0f}" if is_cuda else "", gpu])
    print(f"  -> {out} güncellendi")


if __name__ == "__main__":
    main()
