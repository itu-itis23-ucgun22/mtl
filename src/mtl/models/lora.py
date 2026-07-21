"""LoRA (Low-Rank Adaptation, Hu et al. 2021) — donuk ViT gövdesini UCUZA uyarlamak (Faz 2).

Adaptasyon ekseni (donuk → LoRA → full) için orta nokta. Her hedef Linear'a düşük-rütbeli bir ek
yol (A: d→r, B: r→d) takılır; SADECE A,B eğitilir, taban ağırlık DONUK kalır:

    y = W·x  (donuk)  +  (alpha/r)·B(A(x))   (eğitilebilir, r « d)

Neden bu tez için doğru araç:
  - ViT-B'nin ~86M'i donuk kalır → eğitilen yalnız ~yüz binler mertebesinde LoRA parametresi
    (12 GB VRAM'e sığar; full fine-tune'dan çok daha hafif).
  - Deneme 9 TAHMİNİNİ test eder: çözüldüğünde EN ÇOK MAE kazanmalı (donukken semantik yoktu),
    DINOv2 en azını (donukken zaten tavana yakın) → sıralama değişmeli.

⚠️ Cache KULLANILAMAZ: LoRA taban trunk'ı değiştirir (her adım farklı çıktı) → feature-caching
geçersiz. Normal `scripts/train.py` ile koş (ROADMAP Faz 2 notu). Ayrıca donuk rejimin aksine
backbone aktivasyonları backprop için saklanır → VRAM daha yüksek (batch'i düşürmen gerekebilir).
"""
from __future__ import annotations

import math

from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Bir `nn.Linear`'ı sarar: taban DONUK, yanına eğitilebilir düşük-rütbeli ek yol.

    B sıfırla başlatılır → başlangıçta ek yol tam olarak 0, yani modül birebir tabana eşittir
    (no-op). Böylece eğitim, pretrained ağırlıktan sapmadan stabil başlar.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank pozitif olmalı, {rank} verildi")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # taban donuk; yalnız A,B eğitilir
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)  # başlangıçta ek yol = 0 → base ile birebir
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(self.dropout(x)))


# hedef adı -> (blok içindeki ebeveyn modül, Linear özniteliği). timm ViT Block yapısı:
#   blk.attn.qkv / blk.attn.proj (attention) · blk.mlp.fc1 / blk.mlp.fc2 (MLP)
_TARGET_PATHS = {
    "qkv": ("attn", "qkv"),
    "proj": ("attn", "proj"),
    "fc1": ("mlp", "fc1"),
    "fc2": ("mlp", "fc2"),
}


def apply_lora_to_vit(
    vit: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: tuple[str, ...] = ("qkv", "proj"),
    last_n_blocks: int = -1,
) -> int:
    """timm ViT bloklarındaki hedef Linear'ları `LoRALinear` ile sarar (yerinde).

    vit: timm ViT — `.blocks` listesi; her blokta `.attn.qkv/.attn.proj/.mlp.fc1/.mlp.fc2`.
    targets: sarılacak Linear'lar (qkv/proj/fc1/fc2). Varsayılan attention (qkv+proj) = kanonik LoRA.
    last_n_blocks: yalnız son N bloğa uygula (-1 = tüm bloklar).
    Döndürür: sarılan Linear sayısı (0 ise config/mimari uyuşmuyordur → çağıran uyarmalı).
    """
    for name in targets:
        if name not in _TARGET_PATHS:
            raise ValueError(f"Bilinmeyen LoRA hedefi '{name}'. Seçenekler: {sorted(_TARGET_PATHS)}")
    blocks = vit.blocks
    n = len(blocks) if last_n_blocks < 0 else min(last_n_blocks, len(blocks))
    wrapped = 0
    for blk in blocks[len(blocks) - n:]:
        for name in targets:
            parent_name, attr = _TARGET_PATHS[name]
            parent = getattr(blk, parent_name, None)
            if parent is None or not hasattr(parent, attr):
                continue  # bu mimaride yok (ör. farklı MLP adı) → atla
            linear = getattr(parent, attr)
            if not isinstance(linear, nn.Linear):
                continue
            setattr(parent, attr, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
            wrapped += 1
    return wrapped
