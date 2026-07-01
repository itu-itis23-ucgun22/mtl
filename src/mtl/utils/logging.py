"""Minimal stdout + CSV step logger - deliberately no wandb/tensorboard
dependency for v1 (noted as a future optional integration in README)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


class CsvLogger:
    def __init__(self, out_dir: str, run_name: str):
        self.path = Path(out_dir) / f"{run_name}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists()

    def log(self, step: int, values: Dict[str, float]) -> None:
        row = {"step": step, **values}
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
        print(f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in values.items()))
