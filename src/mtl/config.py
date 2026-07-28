"""Config dataclasses shared by local CPU smoke runs and Colab/Kaggle GPU runs.

The same YAML schema drives both; only field *values* differ between
configs/smoke_cpu.yaml and configs/train_colab_gpu.yaml.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    ann_file: str = "data/coco_subset/annotations/instances_train_subset.json"
    img_dir: str = "data/coco_subset/images/train"
    val_ann_file: str = "data/coco_subset/annotations/instances_val_subset.json"
    val_img_dir: str = "data/coco_subset/images/val"
    n_images: Optional[int] = None
    img_size: int = 512
    num_workers: int = 0


@dataclass
class ModelConfig:
    backbone_name: str = "resnet50"
    pretrained: bool = True
    trainable_backbone_layers: int = 3
    cls_head_tap: str = "fpn_p5"  # or "backbone_body"
    # --- Görev-özel neck (Faz 3 ablasyon ekseni) ---
    # Segmentation decoder — "bağlam vs maliyet" ablasyonu:
    #   "fcn"    = düz FCN (VARSAYILAN; tüm mevcut sonuçlar bununla), bağlam yok
    #   "aspp"   = ASPP/DeepLabv3, çok-ölçekli bağlam (ağır)
    #   "lraspp" = LR-ASPP/MobileNetV3, hafif global bağlam (ucuz, edge-dostu)
    seg_neck: str = "fcn"
    # Faz 3 "task-interference / task-native neck" ekseni. Değerler:
    #   "shared"             -> tek PAYLAŞILAN SFP üç head'i besler (VARSAYILAN; tüm sonuçlar bununla).
    #   "per_task_identical" -> her göreve KENDİ (yapıca özdeş) SFP neck'i; donuk trunk paylaşılır.
    #                           Tek değişken = neck paylaşımı -> saf interference probu.
    #   "task_native"        -> her göreve NATIVE neck: det=SFP (piramit, zorunlu), seg=ASPP (dense
    #                           context, HAM trunk'tan; SFP yok), cls=GAP+Linear (HAM trunk'tan).
    #                           "göreve özel mimari neck kazandırır mı" (çok-değişkenli; RESULTS'a not).
    # Hepsi donuk trunk üstünde -> cache geçerli. per_task_* yalnız ViT gövdeli (trunk_forward'lı) backbone.
    neck_mode: str = "shared"
    # --- LoRA (Faz 2 adaptasyon ekseni; yalnız ViT gövdeli backbone'lar) ---
    lora: bool = False              # True: ViT gövdesine LoRA adaptörleri tak (taban donuk kalır)
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "qkv,proj"  # virgülle ayrık: qkv/proj/fc1/fc2 (varsayılan attention)
    lora_blocks: int = -1           # son N transformer bloğuna uygula (-1 = tümü)


@dataclass
class LossConfig:
    det_cls: float = 1.0
    det_box: float = 1.0
    seg: float = 1.0
    cls: float = 0.5
    # Faz 3 ablasyonu: True -> sabit ağırlıklar yerine ÖĞRENİLEN belirsizlik ağırlıkları
    # (Kendall et al. 2018). Yukarıdaki det_cls/.../cls değerleri o zaman YOK SAYILIR.
    adaptive: bool = False


@dataclass
class TrainConfig:
    device: str = "cpu"
    batch_size: int = 2
    epochs: int = 1
    max_steps: Optional[int] = None
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = False
    seed: int = 42
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    run_name: str = "run"
    checkpoint_every_steps: Optional[int] = None  # mid-epoch checkpoint cadence; None = only at epoch end


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _apply_section(section, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(section, key):
            raise ValueError(f"Unknown config key '{key}' for {type(section).__name__}")
        setattr(section, key, value)


def _apply_override(cfg: Config, dotted_key: str, raw_value: str) -> None:
    section_name, _, field_name = dotted_key.partition(".")
    if not field_name:
        raise ValueError(f"Override '{dotted_key}' must be of the form section.field=value")
    section = getattr(cfg, section_name)
    current = getattr(section, field_name)
    value = yaml.safe_load(raw_value)
    if current is not None and not isinstance(value, type(current)):
        value = type(current)(value)
    setattr(section, field_name, value)


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """Build a Config from a YAML file plus optional CLI overrides.

    overrides keys are dotted section.field strings, e.g. {"train.device": "cuda"}.
    """
    cfg = Config()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for section_name in ("data", "model", "loss", "train"):
            if section_name in raw:
                _apply_section(getattr(cfg, section_name), raw[section_name])
    for key, value in (overrides or {}).items():
        _apply_override(cfg, key, str(value))
    return cfg


def apply_smoke_overrides(cfg: Config) -> Config:
    """Force a tiny, fast, CPU-safe run regardless of what the loaded config said."""
    cfg.train.max_steps = 2
    cfg.train.epochs = 1
    cfg.data.num_workers = 0
    if cfg.data.n_images is None or cfg.data.n_images > 20:
        cfg.data.n_images = 20
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
