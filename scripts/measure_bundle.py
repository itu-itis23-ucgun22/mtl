"""BUNDLE VERİMİ: 3 görevi '3 AYRI model' ile yapmanın BİRLEŞİK maliyeti vs '1 paylaşılan omurga + 3 head'.

Konuşlandırma motivasyonunun (uçak/edge: 3 ayrı küçük model mi, tek paylaşılan omurga mı?) SAYISAL kanıtı.
"aynı anda 3 görev" = bir görüntü için 3 çıktının HEPSİ gerekir → tek hızlandırıcıda sıralı koşum →
gecikmeler TOPLANIR; 3 model bellekte birlikte durur → parametre ve tepe-bellek TOPLANIR.

⚠️ Verim yalnız MİMARİYE bağlıdır (eğitilmiş ağırlığa değil) → checkpoint YÜKLENMEZ (key-mismatch riski yok).
Her model batch=1 uçtan-uca, bizim measure_efficiency'mizle (warmup+iter, cuda sync) ölçülür = A100 tablosuyla
aynı metodoloji. Metrikler ayrı ölçüldü (eval.py); bu script yalnız BİRLEŞİK VERİMİ verir.

Model türleri (spec):
  mtl:CONFIG          bizim MultiTaskModel (config'ten mimari). Tek-görev uzmanı da tüm head'leri taşır
                      → full-model küçük bir ÜST sınır; backbone-only da raporlanır (adil replikasyon maliyeti).
  segformer:HF_ID     HF SegformerForSemanticSegmentation (gerçek tek-görev, ör. nvidia/mit-b2)
  torchvision:NAME    torchvision detektör: fasterrcnn | retinanet | fcos (gerçek tek-görev)

Örnekler:
  # 3 ayrı ResNet uzmanı (edge, aynı omurga) vs paylaşılan ResNet multi-task:
  python scripts/measure_bundle.py \
      --multitask mtl:configs/train_colab_resnet_frozen.yaml \
      --model mtl:configs/train_colab_resnet_det_ft.yaml \
      --model mtl:configs/train_colab_resnet_seg_ft.yaml \
      --model mtl:configs/train_colab_resnet_cls_ft.yaml

  # heterojen "her göreve en iyi model" bundle vs paylaşılan DINOv2 multi-task:
  python scripts/measure_bundle.py \
      --multitask mtl:configs/train_colab_dinov2_frozen.yaml \
      --model torchvision:fasterrcnn \
      --model segformer:nvidia/mit-b2 \
      --model mtl:configs/train_colab_resnet_cls_ft.yaml
"""
from __future__ import annotations

import argparse

import torch
from torch import nn

from mtl.config import load_config
from mtl.models.multitask_model import MultiTaskModel
from mtl.utils.bench import count_params, measure_efficiency
from mtl.utils.device import resolve_device
from mtl.utils.results import append_result

NUM_CLASSES = 80  # COCO subset: 80 sınıf (seg = +1 bg, cls = 80 çok-etiketli)


class _DetWrap(nn.Module):
    """torchvision detektörü tensör girdisine sarar: model(x:(B,3,H,W)) -> model([img,...])
    (measure_efficiency `module(x)` çağırır; torchvision detektörleri liste-girdi ister)."""

    def __init__(self, det: nn.Module):
        super().__init__()
        self.det = det

    def forward(self, x: torch.Tensor):
        return self.det([img for img in x])


def build_model(spec: str, device: torch.device):
    """spec -> (model, backbone_or_None, img_size, label). Ağırlık YÜKLENMEZ (verim mimariye bağlı)."""
    kind, _, arg = spec.partition(":")
    if kind == "mtl":
        cfg = load_config(arg)
        m = MultiTaskModel(
            backbone_name=cfg.model.backbone_name, pretrained=False,
            trainable_backbone_layers=cfg.model.trainable_backbone_layers,
            det_num_classes=NUM_CLASSES, seg_num_classes=NUM_CLASSES + 1, cls_num_labels=NUM_CLASSES,
            lora=cfg.model.lora, lora_rank=cfg.model.lora_rank, lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout, lora_targets=cfg.model.lora_targets,
            lora_blocks=cfg.model.lora_blocks, adaptive_loss=cfg.loss.adaptive,
            seg_neck=cfg.model.seg_neck, neck_mode=cfg.model.neck_mode, det_neck=cfg.model.det_neck,
            multilayer_taps=cfg.model.multilayer_taps, det_box_loss=cfg.model.det_box_loss,
        ).to(device)
        return m, m.backbone, cfg.data.img_size, f"mtl:{cfg.model.backbone_name}"
    if kind == "segformer":
        from transformers import SegformerForSemanticSegmentation
        m = SegformerForSemanticSegmentation.from_pretrained(
            arg, num_labels=NUM_CLASSES + 1, ignore_mismatched_sizes=True
        ).to(device)
        return m, None, 512, f"segformer:{arg.split('/')[-1]}"
    if kind == "torchvision":
        import torchvision
        builders = {
            "fasterrcnn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
            "retinanet": torchvision.models.detection.retinanet_resnet50_fpn,
            "fcos": torchvision.models.detection.fcos_resnet50_fpn,
        }
        det = builders[arg](weights="DEFAULT").to(device)
        return _DetWrap(det), None, 512, f"torchvision:{arg}"
    raise SystemExit(f"bilinmeyen model türü: {kind!r} (mtl|segformer|torchvision)")


def measure(spec: str, device: torch.device, img_size_override: int | None):
    """Bir modelin full-model (+ varsa backbone-only) batch=1 verimini ölç."""
    model, backbone, native, label = build_model(spec, device)
    # mtl backbone girdi boyutu MİMARİYE bağlı (patch14 DINOv2 → 518; 512 bölünmez) → override'ı
    # YOKSAY, config'in native boyutunu kullan. Esnek dış modeller (segformer/torchvision) override alır.
    kind = spec.partition(":")[0]
    img_size = native if (kind == "mtl" or not img_size_override) else img_size_override
    full = measure_efficiency(model, device, img_size=img_size, batch=1)
    back = measure_efficiency(backbone, device, img_size=img_size, batch=1) if backbone is not None else None
    del model, backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "img_size": img_size, "full": full, "backbone": back}


def _fmt(e: dict) -> str:
    mem = f"{e['peak_mem_MB']:.0f}MB" if e["peak_mem_MB"] == e["peak_mem_MB"] else "-"
    return f"{e['params_M']:6.1f}M  {e['latency_ms']:7.2f}ms  {e['fps']:6.1f}fps  {mem:>8}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, metavar="SPEC",
                   help="bundle'daki uzman model (tekrarlanabilir). Ör: mtl:configs/x.yaml, segformer:nvidia/mit-b2, torchvision:fasterrcnn")
    p.add_argument("--multitask", default=None, metavar="SPEC",
                   help="kıyas için tek paylaşılan model (genelde mtl:CONFIG). Bundle ile yan yana basılır.")
    p.add_argument("--img-size", type=int, default=512,
                   help="tüm modellere verilen girdi boyu (edge kamera girdisi); her model içeride kendi yeniden-boyutlamasını yapar. 0 = her modelin native'i")
    p.add_argument("--device", default="cuda")
    p.add_argument("--results-csv", default="runs/results.csv")
    args = p.parse_args()

    device = resolve_device(args.device)
    override = None if args.img_size == 0 else args.img_size

    print(f"\n{'='*78}\n BUNDLE VERİMİ (batch=1, {device.type}, girdi {args.img_size or 'native'})"
          f"\n{'='*78}")
    print(f"{'model':<26}{'params':>9}{'gecikme':>11}{'FPS':>9}{'bellek':>10}")
    print("-" * 78)

    results = [measure(s, device, override) for s in args.model]
    tot_params = tot_lat = tot_mem = 0.0
    mem_valid = True
    for r in results:
        e = r["full"]
        print(f"{r['label']:<26}{_fmt(e)}")
        tot_params += e["params_M"]
        tot_lat += e["latency_ms"]
        if e["peak_mem_MB"] == e["peak_mem_MB"]:
            tot_mem += e["peak_mem_MB"]
        else:
            mem_valid = False
    bundle_fps = 1000.0 / tot_lat
    mem_str = f"{tot_mem:.0f}MB" if mem_valid else "-"
    print("-" * 78)
    print(f"{'BUNDLE (Σ = 3 ayrı model)':<26}{tot_params:6.1f}M  {tot_lat:7.2f}ms  {bundle_fps:6.1f}fps  {mem_str:>8}")

    if args.multitask:
        mt = measure(args.multitask, device, override)
        e = mt["full"]
        print(f"{'PAYLAŞILAN (1 model)':<26}{_fmt(e)}")
        print("-" * 78)
        ratio = (f"  → BUNDLE / PAYLAŞILAN oranı:  params {tot_params/e['params_M']:.2f}×   "
                 f"gecikme {tot_lat/e['latency_ms']:.2f}×")
        if mem_valid and e["peak_mem_MB"] == e["peak_mem_MB"]:
            ratio += f"   bellek {tot_mem/e['peak_mem_MB']:.2f}×"
        print(ratio)
        append_result(args.results_csv, {
            "run_name": "bundle_multitask", "backbone": mt["label"], "trainable_layers": "-",
            "checkpoint": "-", "split": "bundle-eff", "step": 0,
            "params_M": round(e["params_M"], 2), "latency_ms": round(e["latency_ms"], 2),
            "fps": round(e["fps"], 1),
            "peak_mem_MB": round(e["peak_mem_MB"], 0) if e["peak_mem_MB"] == e["peak_mem_MB"] else "",
        })

    # backbone-only kırılım (mtl modelleri için) — full-model'deki head "üst sınır"ını görünür kılar
    backs = [r for r in results if r["backbone"] is not None]
    if backs:
        print(f"\n  backbone-only kırılım (replikasyonun ADİL maliyeti; head'ler iki rejimde de ortak):")
        for r in backs:
            print(f"    {r['label']:<24}{_fmt(r['backbone'])}")

    append_result(args.results_csv, {
        "run_name": "bundle_separate", "backbone": "+".join(r["label"] for r in results),
        "trainable_layers": "-", "checkpoint": "-", "split": "bundle-eff", "step": 0,
        "params_M": round(tot_params, 2), "latency_ms": round(tot_lat, 2),
        "fps": round(bundle_fps, 1), "peak_mem_MB": round(tot_mem, 0) if mem_valid else "",
    })
    print(f"\n[results] {args.results_csv}'ye eklendi (bundle_separate + bundle_multitask).")


if __name__ == "__main__":
    main()
