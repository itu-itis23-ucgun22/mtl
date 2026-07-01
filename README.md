# MTL — Multi-Task CV with a Shared ResNet50 Backbone

One shared ResNet50+FPN backbone, jointly trained for three tasks on a COCO
subset:

- **Detection** — RetinaNet head (Focal Loss + smooth-L1)
- **Semantic segmentation** — FCN head on the finest FPN level
- **Multi-label classification** — GAP+FC head on the coarsest FPN level

See `.claude/plans` (or ask the assistant) for the full design rationale.
Short version: v1 deliberately avoids full Mask R-CNN / instance
segmentation, because torchvision's `MaskRCNN` bundles RPN+RoIAlign+mask
head inside its own forward pass in a way that isn't a clean extension
point for bolting on a 4th (classification) task. RetinaNet exposes a
`backbone -> feature dict -> head` boundary instead, which three heads can
share safely. Instance segmentation (`MaskRCNN`) is a documented v2 upgrade
in `src/mtl/models/maskrcnn_v2.py`, reusing the same backbone.

## Hardware note

This machine has no NVIDIA GPU (AMD Radeon RX 550X, no usable CUDA/ROCm).
The workflow is split:

- **Local (this machine, CPU only)**: write and debug code, run tiny smoke
  tests (`--smoke`, a couple of images, 1-2 steps) to catch bugs before
  ever touching a GPU.
- **Colab / Kaggle (free NVIDIA GPU)**: real training runs, via
  `notebooks/colab_train.ipynb`.

The same `scripts/train.py` and YAML configs drive both — only `device` and
dataset size differ.

## Setup (local)

```
conda create -n mtl python=3.11
conda activate mtl
pip install -r requirements-local.txt
pip install -e .
```

## Quick smoke test (no dataset needed to be huge - use the 20-image fixture path)

```
python scripts/train.py --config configs/smoke_cpu.yaml --smoke
pytest tests/
```

## Building the COCO subset

Download COCO 2017 `train2017`/`val2017` images + `annotations_trainval2017.zip`
separately (not included here — too large), then:

```
python scripts/prepare_coco_subset.py \
    --ann-file /path/to/instances_train2017.json \
    --out data/coco_subset/annotations/instances_train_subset.json \
    --n-images 22500

python scripts/prepare_coco_subset.py \
    --ann-file /path/to/instances_val2017.json \
    --out data/coco_subset/annotations/instances_val_subset.json \
    --n-images 2000
```

Then download only the images the subset actually references (via each
image's `coco_url` field) - much cheaper than the full `train2017.zip`
(~18GB) / `val2017.zip` (~1GB):

```
python scripts/download_subset_images.py \
    --ann-file data/coco_subset/annotations/instances_train_subset.json \
    --out-dir data/coco_subset/images/train

python scripts/download_subset_images.py \
    --ann-file data/coco_subset/annotations/instances_val_subset.json \
    --out-dir data/coco_subset/images/val
```

## Training on Colab/Kaggle GPU

Open `notebooks/colab_train.ipynb` — it installs dependencies, pulls the
repo, builds the COCO subset, and runs
`python scripts/train.py --config configs/train_colab_gpu.yaml`.

## Project layout

```
src/mtl/
  config.py            # YAML + CLI-override config
  models/               # backbone, 3 heads, MultiTaskModel, v2 stub
  losses/joint_loss.py  # fixed-weight combined loss
  datasets/             # COCO subset builder, dataset, transforms, collate
  engine/               # train loop, evaluation, checkpointing
  utils/                # device resolution, seeding, logging
configs/                 # smoke_cpu.yaml, train_colab_gpu.yaml
scripts/                 # train.py, eval.py, prepare_coco_subset.py
notebooks/colab_train.ipynb
tests/                   # pytest smoke + shape tests
```
