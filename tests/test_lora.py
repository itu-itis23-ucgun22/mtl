"""LoRALinear ve apply_lora_to_vit için testler (Faz 2 adaptasyon ekseni).
Ağ/GPU gerektirmez; sahte bir timm-benzeri ViT üstünde CPU'da saniyeler içinde koşar."""
from __future__ import annotations

import torch
from torch import nn

from mtl.models.lora import LoRALinear, apply_lora_to_vit


# --- timm ViT Block yapısını taklit eden minimal sahte gövde (ağsız) ---
class _FakeAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)


class _FakeMlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)


class _FakeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _FakeAttn(dim)
        self.mlp = _FakeMlp(dim)


class _FakeViT(nn.Module):
    def __init__(self, dim=16, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(depth)])


def test_lora_linear_is_noop_at_init():
    """B sıfır başlatıldığı için başlangıçta LoRALinear tam olarak tabana eşit olmalı."""
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_linear_only_ab_trainable():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4)
    assert all(not p.requires_grad for p in lora.base.parameters())  # taban donuk
    assert lora.lora_a.weight.requires_grad and lora.lora_b.weight.requires_grad


def test_lora_linear_changes_after_grad_step():
    """Bir optimizasyon adımından sonra (B artık sıfır değil) çıktı tabandan sapmalı."""
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 8)
    opt = torch.optim.SGD([p for p in lora.parameters() if p.requires_grad], lr=0.1)
    lora(x).sum().backward()
    opt.step()
    assert not torch.allclose(lora(x), base(x), atol=1e-6)


def test_apply_lora_wraps_attention_targets():
    vit = _FakeViT(dim=16, depth=3)
    n = apply_lora_to_vit(vit, rank=4, targets=("qkv", "proj"))
    assert n == 3 * 2  # her blokta qkv + proj
    for blk in vit.blocks:
        assert isinstance(blk.attn.qkv, LoRALinear)
        assert isinstance(blk.attn.proj, LoRALinear)
        assert isinstance(blk.mlp.fc1, nn.Linear)  # hedeflenmedi → sarılmadı


def test_apply_lora_last_n_blocks_only():
    vit = _FakeViT(dim=16, depth=4)
    n = apply_lora_to_vit(vit, rank=4, targets=("qkv",), last_n_blocks=2)
    assert n == 2  # yalnız son 2 blok, blok başına 1 hedef
    assert isinstance(vit.blocks[0].attn.qkv, nn.Linear)      # ilk bloklar dokunulmadı
    assert isinstance(vit.blocks[-1].attn.qkv, LoRALinear)    # son blok sarıldı


def test_apply_lora_mlp_targets():
    vit = _FakeViT(dim=16, depth=2)
    n = apply_lora_to_vit(vit, rank=4, targets=("fc1", "fc2"))
    assert n == 2 * 2
    for blk in vit.blocks:
        assert isinstance(blk.mlp.fc1, LoRALinear) and isinstance(blk.mlp.fc2, LoRALinear)
        assert isinstance(blk.attn.qkv, nn.Linear)  # attention hedeflenmedi
