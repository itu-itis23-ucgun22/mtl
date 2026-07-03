"""runs/results.csv'yi okuyup ResNet-vs-DINO markdown karşılaştırma tablosu üretir.

    python scripts/compare_results.py                      # stdout'a bas
    python scripts/compare_results.py --out compare.md     # dosyaya da yaz

İki tablo verir:
  1. Tüm koşular (her eval satırı).
  2. Omurga başına EN İYİ metrik özeti (satır=metrik, kolon=omurga) - "tek bakışta
     ResNet vs DINO" karşılaştırması için. (EXPERIMENTS.md'ye kopyalanabilir.)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

METRIC_KEYS = ["detection_mAP", "seg_mIoU", "cls_mAP", "cls_F1"]


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fmt(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return str(value)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def all_runs_table(rows: List[Dict[str, str]]) -> str:
    cols = ["timestamp", "run_name", "backbone", "trainable_layers", "step", *METRIC_KEYS]
    body = [[_fmt(r.get(c, "")) if c in METRIC_KEYS else r.get(c, "") for c in cols] for r in rows]
    return _md_table(cols, body)


def best_per_backbone_table(rows: List[Dict[str, str]]) -> str:
    """satır=metrik, kolon=omurga; hücre=o omurganın koşuları arasındaki en iyi (max) değer."""
    backbones = sorted({r["backbone"] for r in rows})
    headers = ["metric", *backbones]
    body = []
    for metric in METRIC_KEYS:
        cells = [metric]
        for bb in backbones:
            vals = []
            for r in rows:
                if r["backbone"] == bb:
                    try:
                        vals.append(float(r.get(metric, "")))
                    except (ValueError, TypeError):
                        pass
            cells.append(f"{max(vals):.4f}" if vals else "-")
        body.append(cells)
    return _md_table(headers, body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="runs/results.csv")
    parser.add_argument("--out", help="markdown çıktısını bu dosyaya da yaz")
    args = parser.parse_args()

    if not Path(args.results_csv).exists():
        raise SystemExit(
            f"{args.results_csv} yok. Önce scripts/eval.py ile en az bir koşu değerlendir."
        )

    rows = _read_rows(args.results_csv)
    if not rows:
        raise SystemExit(f"{args.results_csv} boş.")

    md = (
        "## Tüm koşular\n\n" + all_runs_table(rows)
        + "\n\n## Omurga başına en iyi (ResNet vs DINO)\n\n" + best_per_backbone_table(rows)
        + "\n"
    )
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n[compare] {args.out} yazıldı.")


if __name__ == "__main__":
    main()
