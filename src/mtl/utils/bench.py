"""Omurga çıkarım maliyeti ölçümü (parametre · gecikme · FPS · tepe bellek).

Tek kaynak: hem `scripts/benchmark_latency.py` (tek başına kıyas) hem `scripts/eval.py`
(metriklerle aynı çıktıda) bunu kullanır — böylece iki yerde AYNI şey ölçülür.

Ölçülen: verilen modülün ileri-geçişi (genelde `model.backbone` = backbone+neck, üç görevin
PAYLAŞTIĞI kısım). Isınma + senkronizasyon + çok-iter ortalaması ile güvenilir zamanlama.
"""
from __future__ import annotations

import time
from typing import Dict

import torch
from torch import nn


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def _time_forward(module: nn.Module, x: torch.Tensor, iters: int, is_cuda: bool) -> float:
    """iters kez ileri-geçiş; toplam saniye (GPU senkronizasyonu dahil)."""
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        module(x)
    if is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def measure_efficiency(
    module: nn.Module,
    device: torch.device,
    img_size: int,
    batch: int = 1,
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, float]:
    """`module`'ün çıkarım maliyetini ölç. Dönen dict: params_M · latency_ms · fps · peak_mem_MB.

    batch=1 -> gerçek-zaman/edge gecikmesi (uçak senaryosu). module eval moduna alınır ve
    ölçüm sonrası eski moduna döndürülür (çağıranın train/eval durumunu bozmaz).
    """
    is_cuda = device.type == "cuda"
    was_training = module.training
    module.eval()

    x = torch.randn(batch, 3, img_size, img_size, device=device)
    _time_forward(module, x, warmup, is_cuda)  # ısınma (cuDNN autotune, lazy init)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    total_s = _time_forward(module, x, iters, is_cuda)

    per_iter_ms = total_s / iters * 1000.0
    per_img_ms = per_iter_ms / batch
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if is_cuda else float("nan")

    if was_training:
        module.train()

    return {
        "params_M": count_params(module) / 1e6,
        "latency_ms": per_img_ms,
        "fps": 1000.0 / per_img_ms,  # kare/saniye = 1000 / görsel-başı-ms
        "peak_mem_MB": peak_mb,
    }
