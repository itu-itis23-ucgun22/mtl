"""TÜM eğittiğimiz modellerin kalitatif görsellerini TEK komutla üretir (rapor için).

`visualize.py`'ı tematik gruplar halinde çağırır; her grupta **yalnız Drive'da FİİLEN VAR olan**
checkpoint'leri kullanır (eksik olan atlanır → çökme yok). Sonunda bulundu/yok özeti basar.

    python scripts/viz_all.py                         # checkpoints/ , viz/ altına
    python scripts/viz_all.py --checkpoint-dir /content/drive/MyDrive/mtl/checkpoints \
                              --out-root /content/drive/MyDrive/mtl/viz --num-images 6

Notlar:
- Çoklu-görev modelleri (Faz 1/2/3) yan-yana KIYAS figürü olarak gruplanır (GT + her model satırı).
- Tek-görev / referans modelleri (yalnız kendi görevinde geçerli, diğer sütunlar çöp) TEK BAŞINA render edilir.
- Checkpoint adları RESULTS run_name'lerinden türetilmiş **beklenen** adlardır; seninkiler farklıysa
  --checkpoint-dir'deki gerçek adlarla eşleşmeyenler "YOK" olarak raporlanır (yeniden adlandır veya bana ilet).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# key -> (config, beklenen checkpoint basename)
REGISTRY = {
    # FAZ 1 — donuk backbone sweep
    "resnet_frozen": ("train_colab_resnet_frozen.yaml", "colab_resnet_frozen_epoch15.pt"),
    "dino":          ("train_colab_dino.yaml",          "colab_dino_cached_epoch15.pt"),
    "dinov2":        ("train_colab_dinov2.yaml",        "colab_dinov2_epoch15.pt"),
    "clip":          ("train_colab_clip.yaml",          "colab_clip_cached_epoch15.pt"),
    "mae":           ("train_colab_mae.yaml",           "colab_mae_cached_epoch15.pt"),
    "sam":           ("train_colab_sam.yaml",           "colab_sam_cached_epoch15.pt"),
    "ijepa":         ("train_colab_ijepa.yaml",         "colab_ijepa_cached_epoch15.pt"),
    "deit":          ("train_colab_deit.yaml",          "colab_deit_cached_epoch15.pt"),
    # FAZ 2 — LoRA
    "mae_lora":      ("train_colab_mae_lora.yaml",       "colab_mae_lora_epoch15.pt"),
    "dinov2_lora":   ("train_colab_dinov2_lora.yaml",    "colab_dinov2_lora_epoch15.pt"),
    # FAZ 3 — ablasyonlar
    "dinov2_adaptive":        ("train_colab_dinov2_adaptive.yaml",         "colab_dinov2_adaptive_epoch15.pt"),
    "dinov2_taskneck_native": ("train_colab_dinov2_taskneck_native.yaml",  "colab_dinov2_taskneck_native_epoch15.pt"),
    "dinov2_multilayer":      ("train_colab_dinov2_multilayer.yaml",       "colab_dinov2_multilayer_epoch15.pt"),
    "dinov2_segaspp":         ("train_colab_dinov2_segaspp.yaml",          "colab_dinov2_segaspp_epoch15.pt"),
    "dinov2_ciou":            ("train_colab_dinov2_ciou.yaml",             "colab_dinov2_ciou_epoch15.pt"),
    "resnet_segaspp":         ("train_colab_resnet_frozen_segaspp.yaml",         "colab_resnet_frozen_segaspp_epoch15.pt"),
    "resnet_segaspp_detpan":  ("train_colab_resnet_frozen_segaspp_detpan.yaml",  "colab_resnet_frozen_segaspp_detpan_epoch15.pt"),
    # FAZ 3 — tek-görev (kısmi: yalnız kendi görevi geçerli)
    "dinov2_detonly": ("train_colab_dinov2_detonly.yaml", "colab_dinov2_detonly_epoch15.pt"),
    "dinov2_segonly": ("train_colab_dinov2_segonly.yaml", "colab_dinov2_segonly_epoch15.pt"),
    "dinov2_clsonly": ("train_colab_dinov2_clsonly.yaml", "colab_dinov2_clsonly_epoch15.pt"),
    # Referanslar (kısmi: backbone açık + tek görev)
    "resnet_det_ft": ("train_colab_resnet_det_ft.yaml", "colab_resnet_det_ft_epoch15.pt"),
    "resnet_seg_ft": ("train_colab_resnet_seg_ft.yaml", "colab_resnet_seg_ft_epoch15.pt"),
    "resnet_cls_ft": ("train_colab_resnet_cls_ft.yaml", "colab_resnet_cls_ft_epoch15.pt"),
}

# Yan-yana KIYAS grupları (çoklu-görev): (klasör, [key...])
GROUPS = [
    ("1_hero",              ["dinov2", "resnet_frozen", "sam", "mae"]),
    ("2_backbones_rest",    ["dino", "clip", "deit", "ijepa"]),
    ("3_adaptation",        ["mae", "mae_lora", "dinov2", "dinov2_lora"]),
    ("4_resnet_necks",      ["resnet_frozen", "resnet_segaspp", "resnet_segaspp_detpan"]),
    ("5_dinov2_necks",      ["dinov2", "dinov2_taskneck_native", "dinov2_multilayer", "dinov2_segaspp"]),
    ("6_dinov2_loss",       ["dinov2", "dinov2_adaptive", "dinov2_ciou"]),
]

# Tek başına render (kısmi çıktı — yalnız kendi görevi geçerli)
PARTIAL = ["dinov2_detonly", "dinov2_segonly", "dinov2_clsonly",
           "resnet_det_ft", "resnet_seg_ft", "resnet_cls_ft"]

CONFIG_DIR = Path("configs")


def run_visualize(keys, ckpt_dir: Path, out_dir: Path, num_images: int, score_thresh: float) -> bool:
    """Var olan checkpoint'lerle visualize.py'ı çağır. En az 1 model varsa çalıştırır."""
    configs, ckpts = [], []
    for k in keys:
        cfg, ck = REGISTRY[k]
        p = ckpt_dir / ck
        if p.exists():
            configs.append(str(CONFIG_DIR / cfg))
            ckpts.append(str(p))
    if not configs:
        return False
    cmd = [sys.executable, "scripts/visualize.py",
           "--config", *configs, "--checkpoint", *ckpts,
           "--num-images", str(num_images), "--score-thresh", str(score_thresh),
           "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--out-root", default="viz")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--score-thresh", type=float, default=0.3)
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_root = Path(args.out_root)

    # Önce mevcudiyet raporu (hangi checkpoint var/yok) — kullanıcı adları düzeltebilsin
    print(f"\n{'='*70}\n CHECKPOINT MEVCUDİYET ({ckpt_dir})\n{'='*70}")
    found = set()
    for k, (_, ck) in REGISTRY.items():
        ok = (ckpt_dir / ck).exists()
        if ok:
            found.add(k)
        print(f"  [{'VAR' if ok else 'YOK'}] {k:<24} {ck}")
    print(f"\n  → {len(found)}/{len(REGISTRY)} checkpoint bulundu.\n")

    # Kıyas grupları
    for folder, keys in GROUPS:
        present = [k for k in keys if k in found]
        if not present:
            print(f"[grup {folder}] atlandı (hiç checkpoint yok)")
            continue
        print(f"[grup {folder}] modeller: {', '.join(present)}")
        run_visualize(present, ckpt_dir, out_root / folder, args.num_images, args.score_thresh)

    # Tek-görev / referans (tek başına)
    for k in PARTIAL:
        if k not in found:
            continue
        print(f"[partial {k}] tek-başına render (yalnız kendi görevi geçerli)")
        run_visualize([k], ckpt_dir, out_root / "7_partial" / k, args.num_images, args.score_thresh)

    print(f"\nBitti → {out_root} (gruplar: 1_hero .. 6_dinov2_loss, 7_partial/*).")


if __name__ == "__main__":
    main()
