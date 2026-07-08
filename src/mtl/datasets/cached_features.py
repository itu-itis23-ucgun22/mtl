"""Cache'lenmiş trunk feature'ları + hedefleri döndüren dataset (scripts/precompute_features.py çıktısı).

Görüntüyü backbone'dan geçirmek yerine önceden yazılmış trunk feature'ını yükler; hedefler
(kutu/mask/etiket) yine CocoMultiTaskDataset'ten gelir. Böylece eğitim pahalı ViT forward'ını
atlar (scripts/train_cached.py + MultiTaskModel.forward_from_trunk).

Kısıtlar:
  - Backbone DONUK olmalı (trainable_backbone_layers=0) - yoksa trunk her adım değişir, cache geçersiz.
  - base_dataset train=False olmalı (augmentation kapalı) - yoksa flip cache ile eşleşmez.
  - collate: trunk'lar aynı şekilde (embed, h, w) olduğu için mevcut collate_fn ile stack'lenir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir: str, base_dataset: Dataset):
        self.cache_dir = Path(cache_dir)
        self.base = base_dataset
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Feature cache yok: {self.cache_dir}. Önce scripts/precompute_features.py çalıştır."
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        _, target = self.base[index]  # görüntü atılır; yalnızca hedefler (aug yok)
        trunk = np.load(self.cache_dir / f"{index}.npy")  # (embed, h, w) float16
        return torch.from_numpy(trunk).float(), target
