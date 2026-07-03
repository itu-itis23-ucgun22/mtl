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

# Sabit kolon düzeni - satırda eksik anahtar boş bırakılır, fazlası yok sayılır,
# böylece CSV başlığı koşudan koşuya değişmez.
FIELDNAMES = [
    "timestamp",
    "run_name",
    "backbone",
    "trainable_layers",
    "checkpoint",
    "step",
    "detection_mAP",
    "seg_mIoU",
    "cls_mAP",
    "cls_F1",
]


def append_result(csv_path: str, row: Dict[str, object]) -> None:
    """`row`'u csv_path'e ekler; dosya yoksa başlıkla oluşturur.

    `timestamp` verilmemişse otomatik doldurulur. FIELDNAMES dışındaki anahtarlar
    yok sayılır, eksik olanlar boş kalır."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_row = {k: "" for k in FIELDNAMES}
    full_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for k, v in row.items():
        if k in full_row:
            full_row[k] = v

    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(full_row)
