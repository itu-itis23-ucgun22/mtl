"""timm.create_model sarmalayıcısı: ağırlıkları YEREL dosyadan yükleyebilme.

Neden: Colab'dan HuggingFace indirmeleri kopabiliyor (Xet CDN'e erişilemiyor ya da transfer
ortada donuyor). Bu durumda ağırlığı `wget -c` (resume + retry) ile elle indirip bir klasöre
koyup şu env değişkenini set etmek yeterli:

    export MTL_WEIGHTS_DIR=/content/weights
    # ve /content/weights/<timm_model_adi>.safetensors dosyası dursun, ör:
    #   /content/weights/samvit_base_patch16.sa1b.safetensors
    #   /content/weights/vit_base_patch16_224.mae.safetensors

Env set DEĞİLSE ya da dosya yoksa davranış birebir normal timm.create_model'dir (HF'ten indirir).
Yani tamamen ek (additive) - mevcut akışı bozmaz.
"""
from __future__ import annotations

import os

import timm
from torch import nn


def create_timm_model(model_name: str, **kwargs) -> nn.Module:
    """timm.create_model; MTL_WEIGHTS_DIR/<model_name>.safetensors varsa ağırlığı oradan yükler."""
    weights_dir = os.environ.get("MTL_WEIGHTS_DIR")
    if weights_dir and kwargs.get("pretrained"):
        path = os.path.join(weights_dir, f"{model_name}.safetensors")
        if os.path.exists(path):
            print(f"[timm_weights] yerel ağırlık kullanılıyor: {path}")
            kwargs["pretrained_cfg_overlay"] = dict(file=path)
        else:
            print(f"[timm_weights] {path} yok -> HF'ten indirilecek")
    return timm.create_model(model_name, **kwargs)
