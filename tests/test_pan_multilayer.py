"""Faz 3 detection-neck ekseni: PAN (det_neck=pan) + DINOv2 çok-katmanlı aggregation (multilayer_taps).

PAN: base piramidin üstüne bottom-up yol, yalnız detection'a; 5-seviye/256-ch sözleşmesi + seg/cls
etkilenmemesi doğrulanır. Multilayer: trunk (N*embed, h, w) döner (cache formatı rank-4 kalır), neck
kanalları N katmana bölüp seviyelere dağıtır; forward + forward_from_trunk (cache yolu) sözleşmeyi korur.
pretrained=False -> ağ gerekmez. Multilayer testleri timm ister (yoksa atlanır).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mtl.models.multitask_model import MultiTaskModel

NUM_CLASSES = 5
LOSS_KEYS = {"classification", "bbox_regression", "seg_loss", "cls_loss"}


def _targets(batch_size, img_size):
    return [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
            "labels": torch.tensor([0], dtype=torch.int64),
            "sem_mask": torch.zeros(img_size, img_size, dtype=torch.int64),
            "cls_labels": torch.zeros(NUM_CLASSES, dtype=torch.float32),
        }
        for _ in range(batch_size)
    ]


# ---------------- PAN (backbone-agnostik; ResNet ile ağsız) ----------------

def test_pan_neck_shape_contract():
    from mtl.models.pan import PANNeck

    pan = PANNeck(256)
    feats = {k: torch.rand(2, 256, s, s) for k, s in
             (("0", 32), ("1", 16), ("2", 8), ("3", 4), ("pool", 2))}
    out = pan(feats)
    assert list(out.keys()) == ["0", "1", "2", "3", "pool"]
    for k in feats:
        assert out[k].shape == feats[k].shape, f"{k}: PAN boyutu korumadı"
    # girdi mutasyona uğramamalı (paylaşılan modda seg/cls base dict'i görmeye devam etmeli)
    assert out["0"] is feats["0"]  # en ince seviye değişmeden geçer
    assert out["3"] is not feats["3"]  # kaba seviyeler bottom-up ile yeniden hesaplanır


def test_resnet_det_pan_forward():
    model = MultiTaskModel(
        backbone_name="resnet50", pretrained=False,
        det_num_classes=NUM_CLASSES, seg_num_classes=NUM_CLASSES + 1, cls_num_labels=NUM_CLASSES,
        det_neck="pan",
    )
    model.train()
    img = 128
    loss_dict = model(torch.rand(2, 3, img, img), _targets(2, img))
    assert set(loss_dict) == LOSS_KEYS
    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"{k} sonlu değil"


def test_unknown_det_neck_rejected():
    with pytest.raises(ValueError):
        MultiTaskModel(backbone_name="resnet50", pretrained=False, det_neck="bogus")


# ---------------- Multilayer DINOv2 (timm gerekir) ----------------

@pytest.mark.parametrize("taps", [2, 4])
def test_dinov2_multilayer_trunk_and_forward(taps):
    pytest.importorskip("timm")
    from mtl.models.dinov2_backbone import Dinov2Backbone

    img = 14 * 8  # 112 -> patch14, 8x8 grid
    backbone = Dinov2Backbone("dinov2_vitb14", pretrained=False, trainable_blocks=0, multilayer_taps=taps)
    embed = backbone.vit.embed_dim
    trunk = backbone.trunk_forward(torch.rand(1, 3, img, img))
    assert trunk.shape[1] == taps * embed, f"trunk kanalı {trunk.shape[1]} != {taps}*{embed}"
    feats = backbone.neck_forward(trunk)
    assert list(feats.keys()) == ["0", "1", "2", "3", "pool"]
    assert all(f.shape[1] == 256 for f in feats.values())


def test_dinov2_multilayer_multitask_cache_path():
    """Multilayer + cache yolu (forward_from_trunk) 4 loss sözleşmesini korumalı; PAN ile birlikte."""
    pytest.importorskip("timm")
    img = 14 * 8
    model = MultiTaskModel(
        backbone_name="dinov2_vitb14", pretrained=False,
        det_num_classes=NUM_CLASSES, seg_num_classes=NUM_CLASSES + 1, cls_num_labels=NUM_CLASSES,
        det_neck="pan", multilayer_taps=2,
    )
    model.train()
    images = torch.rand(2, 3, img, img)
    # normal forward
    loss_dict = model(images, _targets(2, img))
    assert set(loss_dict) == LOSS_KEYS and all(torch.isfinite(v) for v in loss_dict.values())
    # cache yolu: trunk (2*embed,h,w) -> forward_from_trunk
    trunk = model.backbone.trunk_forward(images)
    loss_cached = model.forward_from_trunk(trunk, (img, img), _targets(2, img))
    assert set(loss_cached) == LOSS_KEYS and all(torch.isfinite(v) for v in loss_cached.values())


def test_multilayer_rejected_on_non_dinov2():
    from mtl.models.backbone import build_backbone

    with pytest.raises(NotImplementedError):
        build_backbone("resnet50", pretrained=False, multilayer_taps=2)
