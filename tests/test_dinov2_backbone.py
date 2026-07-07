"""DINOv2 omurgasının (patch14) DINOv1 ile aynı arayüz/shape sözleşmesini ürettiğini doğrular:
5-seviyeli, 256-kanal feature dict (grid x4/x2/x1/÷2/÷4) ve out_channels=256.

Giriş patch'in 8 katı (8*14=112) seçilir -> 8x8 token gridi; böylece seviye boyutları patch16
DINOv1 testiyle birebir aynı sayılara oturur. Register'lı varyant prefix-token (CLS+register)
kırpmasını da sınar.

pretrained=False -> ağ/internet gerekmez (timm mimariyi yerelde kurar). timm yoksa atlanır.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from mtl.models.dinov2_backbone import Dinov2Backbone


@pytest.mark.parametrize("model_name", ["dinov2_vitb14", "dinov2_vitb14_reg"])
def test_dinov2_backbone_interface(model_name):
    backbone = Dinov2Backbone(model_name=model_name, pretrained=False, trainable_blocks=0)
    assert backbone.out_channels == 256

    img_size = backbone.patch_size * 8  # 112 -> 8x8 stride-16 (nominal) token gridi
    images = torch.rand(2, 3, img_size, img_size)
    feats = backbone(images)

    assert list(feats.keys()) == ["0", "1", "2", "3", "pool"]
    grid = img_size // backbone.patch_size  # level "2" grid boyutu
    expected_hw = {"0": grid * 4, "1": grid * 2, "2": grid, "3": grid // 2, "pool": grid // 4}
    for key, hw in expected_hw.items():
        fmap = feats[key]
        assert fmap.shape[0] == 2
        assert fmap.shape[1] == 256, f"level {key}: kanal {fmap.shape[1]} != 256"
        assert fmap.shape[2] == hw and fmap.shape[3] == hw, (
            f"level {key}: {tuple(fmap.shape[2:])} != {(hw, hw)}"
        )


def test_dinov2_backbone_frozen_by_default():
    backbone = Dinov2Backbone(model_name="dinov2_vitb14", pretrained=False, trainable_blocks=0)
    assert all(not p.requires_grad for p in backbone.vit.parameters())
    assert any(p.requires_grad for p in backbone.out2.parameters())


def test_dinov2_backbone_unfreeze_last_blocks():
    backbone = Dinov2Backbone(model_name="dinov2_vitb14", pretrained=False, trainable_blocks=2)
    assert any(p.requires_grad for p in backbone.vit.blocks[-1].parameters())
    assert all(not p.requires_grad for p in backbone.vit.blocks[0].parameters())


def test_dinov2_backbone_rejects_unknown_name():
    with pytest.raises(ValueError):
        Dinov2Backbone(model_name="dinov2_vitL14", pretrained=False)
