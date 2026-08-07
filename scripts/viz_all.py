"""TÜM eğittiğimiz modellerin kalitatif görsellerini TEK komutla üretir (rapor için).

`visualize.py`'ı tematik gruplar halinde çağırır. Her model için **checkpoint'i OTOMATİK seçer**:
config'in `run_name`'i altında `{run_name}_epoch{N}.pt` ve `{run_name}_step{M}.pt` dosyalarından
**en yüksek global adımlısını** (= SON checkpoint) kullanır. Böylece:
  - tamamlanan koşularda son epoch (ör. epoch15),
  - YARIDA KESİLEN koşularda son step checkpoint'i (epoch15 yoksa) otomatik seçilir.
Eksik model atlanır (çökme yok). Sonunda hangi model → hangi dosya (kaçıncı adım) seçildi raporu basar.

    python scripts/viz_all.py                          # checkpoints/ , viz/
    python scripts/viz_all.py --checkpoint-dir /content/drive/MyDrive/mtl/checkpoints \
                              --out-root /content/drive/MyDrive/mtl/viz --num-images 6

Checkpoint adı config'in run_name'inden gelir → tahmin yok. Drive'da adları elle değiştirdiysen
run_name ile eşleşmeyenler "YOK" raporlanır.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

from mtl.config import load_config

# key -> (config, GERÇEK run_name). run_name Drive'daki dosya adlarından (kesin) → find_latest onu arar.
# Checkpoint OTOMATİK: run_name altındaki epoch/step'lerden en yüksek global adım. Config = mimari (build_model).
REGISTRY = {
    # FAZ 1 — donuk backbone sweep (⚠️ bu Drive klasöründe YOK; başka klasör verilirse çalışır)
    "resnet_frozen": ("train_colab_resnet_frozen.yaml", "colab_resnet_frozen"),
    "dino":          ("train_colab_dino.yaml",          "colab_dino_cached"),
    "dinov2":        ("train_colab_dinov2.yaml",        "colab_dinov2"),
    "clip":          ("train_colab_clip.yaml",          "colab_clip_cached"),
    "mae":           ("train_colab_mae.yaml",           "colab_mae_cached"),
    "sam":           ("train_colab_sam.yaml",           "colab_sam_cached"),
    "ijepa":         ("train_colab_ijepa.yaml",         "colab_ijepa_cached"),
    "deit":          ("train_colab_deit.yaml",          "colab_deit_cached"),
    # FAZ 2 — LoRA
    "mae_lora":      ("train_colab_mae_lora.yaml",      "colab_mae_lora"),
    "dinov2_lora":   ("train_colab_dinov2_lora.yaml",   "colab_dinov2_lora"),
    # FAZ 3 — ablasyonlar (gerçek run_name'ler: cached var/yok tutarsız)
    "dinov2_adaptive":        ("train_colab_dinov2_adaptive.yaml",        "colab_dinov2_adaptive_cached"),
    "dinov2_taskneck_native": ("train_colab_dinov2_taskneck_native.yaml", "colab_dinov2_taskneck_native_cached"),
    "dinov2_multilayer":      ("train_colab_dinov2_multilayer.yaml",      "colab_dinov2_multilayer"),
    "dinov2_ciou":            ("train_colab_dinov2_ciou.yaml",            "colab_dinov2_ciou_cached"),
    "resnet_segaspp":         ("train_colab_resnet_frozen_segaspp.yaml",        "colab_resnet_frozen_segaspp"),
    "resnet_segaspp_detpan":  ("train_colab_resnet_frozen_segaspp_detpan.yaml", "colab_resnet_frozen_segaspp_detpan"),
    # FAZ 3 — tek-görev (kısmi: yalnız kendi görevi geçerli)
    "dinov2_detonly": ("train_colab_dinov2_detonly.yaml", "colab_dinov2_detonly_cached"),
    "dinov2_segonly": ("train_colab_dinov2_segonly.yaml", "colab_dinov2_segonly_cached"),
    "dinov2_clsonly": ("train_colab_dinov2_clsonly.yaml", "colab_dinov2_clsonly_cached"),
    # Referanslar (kısmi: backbone açık + tek görev)
    "resnet_det_ft": ("train_colab_resnet_det_ft.yaml", "colab_resnet_det_ft"),
    "resnet_seg_ft": ("train_colab_resnet_seg_ft.yaml", "colab_resnet_seg_ft"),
    "resnet_cls_ft": ("train_colab_resnet_cls_ft.yaml", "colab_resnet_cls_ft"),
}

# Yan-yana KIYAS grupları. Baseline (colab_dinov2) bu klasörde yok → dinov2_ciou (≈baseline: det/seg/cls
# baseline'a ~eşit) DINOv2 referans satırı olarak kullanılır. Faz 1 grupları başka klasörde otomatik çalışır.
GROUPS = [
    # -- Faz 1 baseline'ları varsa (başka klasör) --
    ("1_hero",           ["dinov2", "resnet_frozen", "sam", "mae"]),
    ("2_backbones_rest", ["dino", "clip", "deit", "ijepa"]),
    # -- bu klasörde mevcut olanlar --
    ("3_dinov2_variants", ["dinov2_ciou", "dinov2_taskneck_native", "dinov2_multilayer", "dinov2_adaptive"]),
    ("4_lora_adaptation", ["dinov2_ciou", "dinov2_lora", "mae_lora"]),
    ("5_resnet_necks",    ["resnet_segaspp", "resnet_segaspp_detpan"]),
]
# Tek başına render (kısmi çıktı — yalnız kendi görevi geçerli)
PARTIAL = ["dinov2_detonly", "dinov2_segonly", "dinov2_clsonly",
           "resnet_det_ft", "resnet_seg_ft", "resnet_cls_ft"]

CONFIG_DIR = Path("configs")


def find_latest(ckpt_dir: Path, run_name: str, spe: int):
    """run_name'e ait epoch/step checkpoint'lerinden EN YÜKSEK global adımlısını döndür.
    step dosyası → adım = M; epoch dosyası → adım = (N+1)*spe. Döner: (Path, adım) veya (None, -1).
    Regex, run_name'i tam eşler → 'colab_dinov2' ile 'colab_dinov2_lora'yı KARIŞTIRMAZ."""
    pat_step = re.compile(rf"^{re.escape(run_name)}_step(\d+)\.pt$")
    pat_epoch = re.compile(rf"^{re.escape(run_name)}_epoch(\d+)\.pt$")
    best, best_step = None, -1
    for p in ckpt_dir.glob(f"{run_name}_*.pt"):
        m = pat_step.match(p.name)
        s = int(m.group(1)) if m else None
        if s is None:
            m = pat_epoch.match(p.name)
            s = (int(m.group(1)) + 1) * spe if m else None
        if s is not None and s > best_step:
            best, best_step = p, s
    return best, best_step


def resolve(key: str, ckpt_dir: Path):
    """key -> (config_path, checkpoint_path, adım) | None (checkpoint yoksa)."""
    cfg_name, run_name = REGISTRY[key]
    cfg_path = CONFIG_DIR / cfg_name
    cfg = load_config(str(cfg_path))
    spe = math.ceil((cfg.data.n_images or 22500) / cfg.train.batch_size) or 5625
    ckpt, step = find_latest(ckpt_dir, run_name, spe)  # GERÇEK run_name (Drive'dan), config'inki değil
    if ckpt is None:
        return None
    return str(cfg_path), str(ckpt), step


def run_visualize(resolved, out_dir: Path, num_images: int, score_thresh: float) -> None:
    configs = [r[0] for r in resolved]
    ckpts = [r[1] for r in resolved]
    cmd = [sys.executable, "scripts/visualize.py", "--config", *configs, "--checkpoint", *ckpts,
           "--num-images", str(num_images), "--score-thresh", str(score_thresh), "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--out-root", default="viz")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--score-thresh", type=float, default=0.3)
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_root = Path(args.out_root)

    # Her model için son checkpoint'i çöz + rapor bas
    print(f"\n{'='*78}\n SEÇİLEN CHECKPOINT'LER (run_name -> en yüksek global adım)\n{'='*78}")
    chosen = {}
    for key in REGISTRY:
        r = resolve(key, ckpt_dir)
        if r:
            chosen[key] = r
            print(f"  [VAR] {key:<24} step {r[2]:>6}  <- {Path(r[1]).name}")
        else:
            print(f"  [YOK] {key:<24} (run_name eşleşen .pt yok)")
    print(f"\n  → {len(chosen)}/{len(REGISTRY)} model için checkpoint bulundu.\n")

    # Kıyas grupları
    for folder, keys in GROUPS:
        present = [chosen[k] for k in keys if k in chosen]
        if not present:
            print(f"[grup {folder}] atlandı (checkpoint yok)")
            continue
        names = ", ".join(k for k in keys if k in chosen)
        print(f"[grup {folder}] modeller: {names}")
        run_visualize(present, out_root / folder, args.num_images, args.score_thresh)

    # Tek-görev / referans (tek başına — diğer sütunlar eğitilmemiş, çöp)
    for key in PARTIAL:
        if key in chosen:
            print(f"[partial {key}] tek-başına render (yalnız kendi görevi geçerli)")
            run_visualize([chosen[key]], out_root / "7_partial" / key, args.num_images, args.score_thresh)

    print(f"\nBitti → {out_root} (1_hero .. 6_dinov2_loss, 7_partial/*).")


if __name__ == "__main__":
    main()
