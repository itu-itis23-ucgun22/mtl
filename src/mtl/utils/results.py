"""Eval sonuçlarını tek bir toplu CSV'ye (varsayılan runs/results.csv) ekler.

Her eval koşusu bir satır: hangi omurga, hangi checkpoint, kaç adım, ve dört metrik.
Böylece ResNet ve DINO koşuları yan yana birikir; scripts/compare_results.py bunu
okuyup markdown karşılaştırma tablosu üretir. (utils/logging.py CsvLogger ile aynı
"başlık bir kez, sonra append" deseni.)
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

# Sabit kolon düzeni - satırda eksik anahtar boş bırakılır, fazlası yok sayılır.
# Kalite metrikleri + verim (params/latency/fps/bellek) tek satırda birlikte durur.
FIELDNAMES = [
    "timestamp",
    "run_name",
    "backbone",
    "trainable_layers",
    "checkpoint",
    "split",
    "step",
    "detection_mAP",
    "seg_mIoU",
    "cls_mAP",
    "cls_F1",
    "params_M",
    "latency_ms",
    "fps",
    "peak_mem_MB",
]


def _to_full_row(row: Dict[str, object]) -> Dict[str, object]:
    full = {k: "" for k in FIELDNAMES}
    for k, v in row.items():
        if k in full:
            full[k] = v
    return full


def append_result(csv_path: str, row: Dict[str, object]) -> None:
    """`row`'u csv_path'e ekler; dosya yoksa başlıkla oluşturur.

    `timestamp` verilmemişse otomatik doldurulur. FIELDNAMES dışındaki anahtarlar yok sayılır,
    eksik olanlar boş kalır. Mevcut dosyanın başlığı eskiyse (yeni kolonlar eklendiyse) dosya
    otomatik GÖÇ ettirilir (eski satırlar korunur, yeni kolonlar boş doldurulur) — böylece
    kolon kayması olmaz ve eski sonuçlar kaybolmaz."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_row = _to_full_row(row)
    full_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing_rows = []
    header_matches = False
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header == FIELDNAMES:
                header_matches = True
            else:
                f.seek(0)
                existing_rows = list(csv.DictReader(f))  # eski başlıkla oku

    if header_matches:
        with open(path, "a", newline="") as f:  # normal ekleme
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(full_row)
    else:  # yeni şemaya göç: eski satırları koru, başlığı güncelle
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for old in existing_rows:
                writer.writerow(_to_full_row(old))
            writer.writerow(full_row)
