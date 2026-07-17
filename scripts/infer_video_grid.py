"""ÇOK-MODEL video karşılaştırması: aynı videoyu N backbone'da koşup tek IZGARA (grid) videosu üretir.

Her karede: aynı görüntü, her model için bir hücre (det + seg overlay + backbone adı + canlı FPS).
Best-practice kıyas: TÜM modeller aynı klip, aynı ön-işleme, aynı --score-thresh -> tek değişken backbone.
FPS overlay'i "hangi model gerçek-zaman uçar" sorusunu görsel yanıtlar (motivasyon: kısıtlı platform).

    python scripts/infer_video_grid.py \
        --config configs/train_colab_dinov2.yaml configs/train_colab_resnet_frozen.yaml ... \
        --checkpoint ckpt_dinov2.pt ckpt_resnet.pt ... \
        --video demo.mp4 --out grid.mp4 --cols 4 --score-thresh 0.3

Notlar:
  - Her model KENDİ img_size'ında ön-işlenir; çizim orijinal çözünürlükte yapılıp hücre boyutuna küçültülür.
  - Tüm modeller aynı anda GPU'da tutulur (8 model ~6 GB; A100 rahat, T4 sınırda -> gerekirse daha az model).
  - pretrained=False -> ağırlık checkpoint'ten, HF indirme yok.
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
    rng = np.random.default_rng(42)
    return (rng.uniform(60, 255, size=(n + 1, 3))).astype(np.uint8)  # BGR


def preprocess(frame_rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    img = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)


def render_tile(frame_bgr, out, size, cat_names, colors, thresh, seg_alpha, no_seg, name, ms):
    """frame_bgr (kopya) üstüne bu modelin det+seg+banner'ını çizer, aynı kareyi döndürür."""
    H, W = frame_bgr.shape[:2]
    sx, sy = W / size, H / size

    if not no_seg:
        seg = out["seg_pred"][0].cpu().numpy().astype(np.uint8)
        seg_big = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = frame_bgr.copy()
        for cls in np.unique(seg_big):
            if cls == 0:
                continue
            overlay[seg_big == cls] = colors[cls].tolist()
        frame_bgr = cv2.addWeighted(overlay, seg_alpha, frame_bgr, 1 - seg_alpha, 0)

    det = out["detections"][0]
    boxes, scores, labels = det["boxes"].cpu(), det["scores"].cpu(), det["labels"].cpu()
    nd = 0
    for i in range(len(boxes)):
        if scores[i] < thresh:
            continue
        x1, y1, x2, y2 = boxes[i].tolist()
        x1, x2, y1, y2 = int(x1 * sx), int(x2 * sx), int(y1 * sy), int(y2 * sy)
        c = tuple(int(v) for v in colors[int(labels[i]) + 1])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame_bgr, f"{cat_names[int(labels[i])]} {scores[i]:.2f}",
                    (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
        nd += 1

    # üst banner: model adı + FPS + kutu sayısı
    bar = frame_bgr[0:34, 0:W].copy()
    frame_bgr[0:34, 0:W] = cv2.addWeighted(bar, 0.35, np.zeros_like(bar), 0.65, 0)
    cv2.putText(frame_bgr, f"{name}  |  {1000.0/ms:4.1f} FPS  |  det {nd}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="grid.mp4")
    p.add_argument("--cols", type=int, default=4, help="ızgara sütun sayısı")
    p.add_argument("--tile-w", type=int, default=480, help="hücre genişliği (px)")
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--no-seg", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--seg-alpha", type=float, default=0.45)
    args = p.parse_args()

    if len(args.config) != len(args.checkpoint):
        raise SystemExit("--config ve --checkpoint aynı sayıda olmalı")

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")

    # kategori isimleri (ilk config; hepsi aynı COCO) + tüm modelleri kur
    ref = load_config(args.config[0])
    ds = CocoMultiTaskDataset(ref.data.val_ann_file, ref.data.val_img_dir, img_size=ref.data.img_size, train=False)
    cat_names = [c["name"] for c in ds.coco.loadCats(ds.cat_ids)]
    colors = class_colors(ds.num_classes)

    models = []
    for cfg_path, ckpt in zip(args.config, args.checkpoint):
        cfg = load_config(cfg_path)
        m = MultiTaskModel(
            backbone_name=cfg.model.backbone_name, pretrained=False,
            trainable_backbone_layers=cfg.model.trainable_backbone_layers,
            det_num_classes=ds.num_classes, seg_num_classes=ds.num_classes + 1,
            cls_num_labels=ds.num_classes,
        ).to(device)
        load_checkpoint(m, optimizer=None, path=ckpt, map_location=str(device))
        m.eval()
        models.append((cfg.model.backbone_name, cfg.data.img_size, m, deque(maxlen=20)))
        print(f"yüklendi: {cfg.model.backbone_name} <- {ckpt}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"video açılamadı: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_models = len(models)
    cols = min(args.cols, n_models)
    rows = (n_models + cols - 1) // cols
    tw = args.tile_w
    th = int(tw * H / W)
    gw, gh = cols * tw, rows * th
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (gw, gh))
    print(f"ızgara: {cols}x{rows} hücre, çıktı {gw}x{gh} @ {fps_in:.0f}fps, {total} kare")

    is_cuda = device.type == "cuda"
    n = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        canvas = np.zeros((gh, gw, 3), dtype=np.uint8)

        for k, (name, size, model, hist) in enumerate(models):
            x = preprocess(frame_rgb, size, device)
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            hist.append((time.perf_counter() - t0) * 1000.0)
            ms = sum(hist) / len(hist)

            tile = render_tile(frame_bgr.copy(), out, size, cat_names, colors,
                               args.score_thresh, args.seg_alpha, args.no_seg, name, ms)
            tile = cv2.resize(tile, (tw, th))
            r, c = divmod(k, cols)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile

        writer.write(canvas)
        n += 1
        if n % 20 == 0:
            print(f"  {n}/{total or '?'} kare")

    cap.release(); writer.release()
    print(f"\nBitti -> {args.out}  ({n} kare, {n_models} model)")


if __name__ == "__main__":
    main()
