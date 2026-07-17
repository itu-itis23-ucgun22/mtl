"""Video üzerinde kalitatif çıkarım: eğitilmiş MTL modelini kare kare koşup det+seg+cls çizer.

Model KARE-BAŞINA (per-image) çalışır — zamansal bilgi yok; video sadece kare dizisi olarak işlenir.
Her kareye: tahmin edilen kutular + semantik maske bindirilir, köşede CANLI FPS + tahmin edilen sınıflar.
Bu, motivasyona (kısıtlı platform / gerçek-zaman algı) doğrudan bağlanan demo çıktısıdır:
FPS overlay'i "bu omurga 30 fps videoyu gerçek-zaman yakalar mı" sorusunu görsel olarak yanıtlar.

    python scripts/infer_video.py --config configs/train_colab_dinov2.yaml \
        --checkpoint checkpoints/colab_dinov2_epoch15.pt \
        --video giris.mp4 --out cikis.mp4 --score-thresh 0.3

Notlar:
  - pretrained=False -> ağırlıklar checkpoint'ten, HF indirme yok.
  - Ön-işleme dataset ile aynı: kareyi (img_size, img_size)'e BILINEAR resize + ImageNet norm;
    tahminler geri orijinal çözünürlüğe ölçeklenir (kutu x*W/size, y*H/size; maske nearest resize).
  - FPS overlay'i YALNIZ model forward süresini ölçer (çizim/IO hariç) -> benchmark_latency ile tutarlı.
  - `--no-seg` sadece detection çizer (daha temiz görünür).
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from mtl.config import load_config
from mtl.datasets.coco_multitask import CocoMultiTaskDataset
from mtl.engine.checkpoint import load_checkpoint
from mtl.models.multitask_model import MultiTaskModel
from mtl.utils.device import resolve_device

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def class_colors(n: int) -> np.ndarray:
    """Sınıf başına sabit BGR renk (index 0 = arka plan). Seed'li -> kareler arası tutarlı."""
    rng = np.random.default_rng(42)
    return (rng.uniform(60, 255, size=(n + 1, 3))).astype(np.uint8)  # BGR, 0..255


def preprocess(frame_rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """Kare (H,W,3 RGB uint8) -> (1,3,size,size) ImageNet-norm tensör (dataset ile aynı)."""
    img = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)


def draw_banner(frame: np.ndarray, lines, org=(10, 10)) -> None:
    """Sol-üste yarı saydam siyah kutu + beyaz metin (okunabilirlik için)."""
    x, y = org
    fh = 24
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for t in lines) + 16
    h = fh * len(lines) + 8
    box = frame[y:y + h, x:x + w].copy()
    frame[y:y + h, x:x + w] = cv2.addWeighted(box, 0.35, np.zeros_like(box), 0.65, 0)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x + 8, y + fh * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True, help="girdi video (mp4 vb.)")
    p.add_argument("--out", default="infer_out.mp4", help="çıktı video yolu")
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--no-seg", action="store_true", help="sadece detection çiz (maske yok)")
    p.add_argument("--max-frames", type=int, default=None, help="test için kare sınırı")
    p.add_argument("--seg-alpha", type=float, default=0.45)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device if torch.cuda.is_available() else "cpu")
    size = cfg.data.img_size

    # Kategori isimleri + sınıf sayısı (COCO ann'dan); img_dir sadece constructor için (görsel yüklenmez).
    ds = CocoMultiTaskDataset(cfg.data.val_ann_file, cfg.data.val_img_dir, img_size=size, train=False)
    cat_names = [c["name"] for c in ds.coco.loadCats(ds.cat_ids)]
    colors = class_colors(ds.num_classes)

    model = MultiTaskModel(
        backbone_name=cfg.model.backbone_name, pretrained=False,
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        det_num_classes=ds.num_classes, seg_num_classes=ds.num_classes + 1,
        cls_num_labels=ds.num_classes,
    ).to(device)
    load_checkpoint(model, optimizer=None, path=args.checkpoint, map_location=str(device))
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"video açılamadı: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
    sx, sy = W / size, H / size
    is_cuda = device.type == "cuda"

    print(f"{args.video}: {W}x{H} @ {fps_in:.0f}fps, {total} kare | model: {cfg.model.backbone_name}")
    ms_hist = deque(maxlen=30)  # yumuşatılmış FPS için
    n = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = preprocess(frame_rgb, size, device)

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        ms_hist.append((time.perf_counter() - t0) * 1000.0)

        # --- segmentasyon maskesi (isteğe bağlı) ---
        if not args.no_seg:
            seg = out["seg_pred"][0].cpu().numpy().astype(np.uint8)          # (size,size), 0=bg
            seg_big = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
            overlay = frame_bgr.copy()
            for cls in np.unique(seg_big):
                if cls == 0:
                    continue
                overlay[seg_big == cls] = colors[cls].tolist()
            frame_bgr = cv2.addWeighted(overlay, args.seg_alpha, frame_bgr, 1 - args.seg_alpha, 0)

        # --- tespit kutuları ---
        det = out["detections"][0]
        boxes, scores, labels = det["boxes"].cpu(), det["scores"].cpu(), det["labels"].cpu()
        ndrawn = 0
        for i in range(len(boxes)):
            if scores[i] < args.score_thresh:
                continue
            x1, y1, x2, y2 = boxes[i].tolist()
            x1, x2, y1, y2 = int(x1 * sx), int(x2 * sx), int(y1 * sy), int(y2 * sy)
            cls = int(labels[i])
            c = tuple(int(v) for v in colors[cls + 1])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, 2)
            tag = f"{cat_names[cls]} {scores[i]:.2f}"
            cv2.putText(frame_bgr, tag, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
            ndrawn += 1

        # --- overlay: FPS + model + top sınıflar ---
        avg_ms = sum(ms_hist) / len(ms_hist)
        cls_pred = out["cls_pred"][0].cpu()
        topv, topi = torch.topk(cls_pred, min(3, len(cls_pred)))
        top = ", ".join(f"{cat_names[int(i)]}" for v, i in zip(topv.tolist(), topi.tolist()) if v > 0.3)
        draw_banner(frame_bgr, [
            f"{cfg.model.backbone_name}",
            f"FPS: {1000.0/avg_ms:5.1f}  ({avg_ms:.1f} ms)",
            f"cls: {top or '-'}   det: {ndrawn}",
        ])

        writer.write(frame_bgr)
        n += 1
        if n % 30 == 0:
            print(f"  {n}/{total or '?'} kare  |  model {1000.0/avg_ms:.1f} FPS")

    cap.release()
    writer.release()
    print(f"\nBitti -> {args.out}  ({n} kare, model ort. {1000.0/(sum(ms_hist)/len(ms_hist)):.1f} FPS)")


if __name__ == "__main__":
    main()
