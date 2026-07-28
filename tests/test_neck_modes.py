"""neck_mode ekseninin (shared | per_task_identical | task_native) wiring'ini doğrular.

Her mod için: (1) train forward doğru 4 loss anahtarını üretir, (2) eval forward doğru shape'li
tahmin döndürür, (3) cached yol (forward_from_trunk) aynı sözleşmeyi korur. task_native'de det=SFP,
seg=ASPP/FCN (ham trunk), cls=GAP (ham trunk) — hepsi tek donuk trunk'tan, cache geçerli.

pretrained=False -> ağ/internet gerekmez (timm mimariyi yerelde kurar). timm yoksa atlanır.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from mtl.models.multitask_model import MultiTaskModel

NUM_CLASSES = 5
IMG_SIZE = 14 * 8  # 112 -> patch14'e bölünebilir, 8x8 token gridi

LOSS_KEYS = {"classification", "bbox_regression", "seg_loss", "cls_loss"}
PRED_KEYS = {"detections", "seg_pred", "cls_pred"}


def _make_batch(batch_size: int = 2):
    images = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE)
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
            "labels": torch.tensor([0], dtype=torch.int64),
            "sem_mask": torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.int64),
            "cls_labels": torch.zeros(NUM_CLASSES, dtype=torch.float32),
        }
        for _ in range(batch_size)
    ]
    return images, targets


def _build(neck_mode: str, seg_neck: str = "fcn"):
    return MultiTaskModel(
        backbone_name="dinov2_vitb14",
        pretrained=False,
        det_num_classes=NUM_CLASSES,
        seg_num_classes=NUM_CLASSES + 1,
        cls_num_labels=NUM_CLASSES,
        seg_neck=seg_neck,
        neck_mode=neck_mode,
    )


@pytest.mark.parametrize("neck_mode", ["per_task_identical", "task_native"])
def test_train_forward_loss_keys(neck_mode):
    model = _build(neck_mode)
    model.train()
    images, targets = _make_batch()
    loss_dict = model(images, targets)
    assert set(loss_dict) == LOSS_KEYS
    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"{neck_mode}/{k} sonlu değil: {v}"


@pytest.mark.parametrize("neck_mode", ["per_task_identical", "task_native"])
def test_eval_forward_pred_shapes(neck_mode):
    model = _build(neck_mode)
    model.eval()
    images, _ = _make_batch()
    with torch.no_grad():
        out = model(images)
    assert set(out) == PRED_KEYS
    assert out["seg_pred"].shape == (2, IMG_SIZE, IMG_SIZE)  # tam çözünürlüğe upsample
    assert out["cls_pred"].shape == (2, NUM_CLASSES)


def test_task_native_uses_raw_trunk_channels():
    """task_native'de seg/cls head'leri HAM trunk (embed_dim) girdiyle kurulur, 256 (SFP) değil."""
    model = _build("task_native")
    embed_dim = model.backbone.vit.embed_dim
    # cls_head.fc girişi embed_dim olmalı (GAP doğrudan trunk'tan)
    assert model.cls_head.fc.in_features == embed_dim
    # det yine SFP -> 256'lık kendi neck'i var
    assert model.det_neck.out_channels == 256
    # task_native'de paylaşılan-mod seg/cls SFP'leri OLUŞMAMALI
    assert not hasattr(model, "seg_neck") and not hasattr(model, "cls_neck")


def test_cached_forward_from_trunk_task_native():
    """Cache akışı: trunk_forward -> forward_from_trunk aynı loss sözleşmesini korumalı."""
    model = _build("task_native", seg_neck="aspp")
    model.train()
    images, targets = _make_batch()
    trunk = model.backbone.trunk_forward(images)
    loss_dict = model.forward_from_trunk(trunk, (IMG_SIZE, IMG_SIZE), targets)
    assert set(loss_dict) == LOSS_KEYS


def test_task_native_rejects_lraspp():
    with pytest.raises(ValueError):
        _build("task_native", seg_neck="lraspp")


def test_per_task_neck_rejects_non_vit():
    with pytest.raises(ValueError):
        MultiTaskModel(backbone_name="resnet50", pretrained=False, neck_mode="task_native")


def test_unknown_neck_mode_rejected():
    with pytest.raises(ValueError):
        _build("bogus")
