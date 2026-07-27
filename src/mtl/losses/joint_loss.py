"""Joint multi-task loss: a fixed weighted sum of the four per-task losses.

Adaptive/uncertainty weighting (Kendall, Gal & Cipolla, "Multi-Task Learning
Using Uncertainty to Weigh Losses", CVPR 2018 - learn per-task log-variance
parameters and weight each loss as L_i / (2*sigma_i^2) + log(sigma_i)) is a
natural v2 improvement, deferred here to keep v1 easy to debug.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from mtl.config import LossConfig

# Loss-dict anahtarları (dört görev bileşeni). Sıra adaptif ağırlık vektörüyle eşleşir.
_TASK_KEYS = ("classification", "bbox_regression", "seg_loss", "cls_loss")


class UncertaintyWeighter(nn.Module):
    """Öğrenilen belirsizlik ağırlıklandırması (Kendall, Gal & Cipolla, CVPR 2018).

    Sabit ağırlıklar (1/1/1/0.5) yerine her göreve ÖĞRENİLEBİLİR bir log-varyans s_i = log σ_i²
    parametresi verir ve toplam loss'u şöyle kurar:
        L = Σ_i [ exp(-s_i)·L_i + s_i ]
    - exp(-s_i) = görevin "kesinliği" (precision); model belirsiz gördüğü görevi otomatik düşürür.
    - +s_i regularizer: σ→∞ kaçışını engeller (aksi halde tüm ağırlıklar sıfıra giderdi).
    - s init 0 → precision 1, log-terim 0 → başlangıçta eşit ağırlık (1/1/1/1), nötr başlangıç.

    Modelin alt-modülü olarak tutulur → parametreleri otomatik olarak optimizer'a ve checkpoint'e
    girer (model.parameters() / model.state_dict() ikisi de kapsar). Keyfi 0.5 elle-ayarını kaldırır.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(len(_TASK_KEYS)))

    def forward(self, loss_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        total = loss_dict[_TASK_KEYS[0]].new_zeros(())
        raw: Dict[str, float] = {}
        for i, k in enumerate(_TASK_KEYS):
            s = self.log_var[i]
            total = total + torch.exp(-s) * loss_dict[k] + s
            raw[k] = loss_dict[k].item()
        raw["total"] = total.item()
        return total, raw

    def weights(self) -> Dict[str, float]:
        """Öğrenilen efektif ağırlıklar exp(-s_i) — raporlama için (hangi görev baskın)."""
        with torch.no_grad():
            return {k: float(torch.exp(-self.log_var[i])) for i, k in enumerate(_TASK_KEYS)}


def combine_losses(loss_dict: Dict[str, Tensor], weights: LossConfig) -> Tuple[Tensor, Dict[str, float]]:
    total = (
        weights.det_cls * loss_dict["classification"]
        + weights.det_box * loss_dict["bbox_regression"]
        + weights.seg * loss_dict["seg_loss"]
        + weights.cls * loss_dict["cls_loss"]
    )
    raw = {k: v.item() for k, v in loss_dict.items()}
    raw["total"] = total.item()
    return total, raw
