"""DINO omurgasının arayüz/shape sözleşmesini doğrular: ResNet+FPN ile aynı 5-seviyeli,
256-kanal feature dict'i (strides 4/8/16/32/64) ürettiğini ve out_channels=256 olduğunu.

pretrained=False -> ağ/internet gerekmez (timm mimariyi yerelde kurar), test_smoke_forward.py
ile aynı offline desen. timm kurulu değilse test atlanır.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from mtl.models.dino_backbone import DinoBackbone

IMG_SIZE = 128  # 128/16 = 8x8 token gridi
EXPECTED_STRIDES = {"0": 4, "1": 8, "2": 16, "3": 32, "pool": 64}


def test_dino_backbone_interface():
    backbone = DinoBackbone(pretrained=False, trainable_blocks=0)
    assert backbone.out_channels == 256

    images = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    feats = backbone(images)

    assert list(feats.keys()) == ["0", "1", "2", "3", "pool"]
    for key, stride in EXPECTED_STRIDES.items():
        fmap = feats[key]
        assert fmap.shape[0] == 2
        assert fmap.shape[1] == 256, f"level {key}: kanal {fmap.shape[1]} != 256"
        expected_hw = IMG_SIZE // stride
        assert fmap.shape[2] == expected_hw and fmap.shape[3] == expected_hw, (
            f"level {key}: {tuple(fmap.shape[2:])} != {(expected_hw, expected_hw)} (stride {stride})"
        )


def test_dino_backbone_frozen_by_default():
    backbone = DinoBackbone(pretrained=False, trainable_blocks=0)
    # trainable_blocks=0 -> ViT gövdesinin tüm parametreleri donuk...
    assert all(not p.requires_grad for p in backbone.vit.parameters())
    # ...ama neck (feature pyramid) eğitilebilir kalmalı.
    assert any(p.requires_grad for p in backbone.out2.parameters())


def test_dino_backbone_unfreeze_last_blocks():
    backbone = DinoBackbone(pretrained=False, trainable_blocks=2)
    # Son 2 blok eğitilebilir olmalı, ilk bloklar hâlâ donuk.
    assert any(p.requires_grad for p in backbone.vit.blocks[-1].parameters())
    assert all(not p.requires_grad for p in backbone.vit.blocks[0].parameters())
