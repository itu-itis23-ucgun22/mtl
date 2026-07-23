# Yerel Kurulum (Colab değil)

Bu proje Colab'da geliştirildi ama tamamen yerelde de koşar. Aşağıdaki adımlar temiz bir
makinede (arkadaş bilgisayarı, GCP VM, lab makinesi) sıfırdan kurulumu anlatır.

## Ön koşullar
| gereksinim | not |
|---|---|
| **Python 3.10+** | 3.11 önerilir |
| **git** | repoyu çekmek için |
| **NVIDIA GPU + sürücü** | opsiyonel ama eğitim için şart. `nvidia-smi` çalışmalı |
| disk | ~15 GB (veri ~8 GB + ağırlıklar + checkpoint'ler) |

> GPU yoksa kurulum yine çalışır (CPU torch) — **eğitim pratik değildir**, sadece kod/test için.

---

## 1. Repoyu çek
```bash
git clone -b dino-backbone https://github.com/itu-itis23-ucgun22/mtl.git
cd mtl
```

## 2. Tek komutla kurulum
**Linux / macOS**
```bash
bash scripts/setup_local.sh
```
**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
```

Script şunları yapar: Python sürümünü doğrular → `.venv` oluşturur → **GPU varsa CUDA torch**
(`requirements-gpu.txt`), yoksa CPU torch (`requirements-local.txt`) kurar → paketi *editable*
kurar (`pip install -e .`, src-layout için şart) → `torch.cuda.is_available()` ile doğrular.

### Elle kurmak istersen
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu.txt  # GPU yoksa: requirements-local.txt
pip install -e .
```

## 3. Doğrula (veri gerekmez)
```bash
pytest tests/ -q
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 4. Veri (COCO alt-kümesi)
Kod `data/coco_subset/` altında şunu bekler:
```
data/coco_subset/
├── images/train/      (~22.500 jpg)
├── images/val/        (~2.000 jpg)
└── annotations/       instances_{train,val,test}_subset.json
```

**Seçenek A — hazır alt-küme varsa (en hızlı).** Drive'daki `zipped.zip` + `annotations/`'ı
kopyala, aç ve yukarıdaki yapıya yerleştir. İstersen başka bir diske koyup symlink ver:
```bash
ln -s /veri/coco_subset data/coco_subset          # Windows: mklink /D
```

**Seçenek B — sıfırdan üret.** COCO 2017 `annotations_trainval2017.zip`'i indir, sonra:
```bash
python scripts/prepare_coco_subset.py --help      # alt-küme JSON'larını üretir
python scripts/download_subset_images.py --help   # sadece gereken görüntüleri indirir
```

---

## 5. Koşum
```bash
# donuk omurga (feature-caching akışı — hızlı)
python scripts/precompute_features.py --config configs/train_colab_mae.yaml --split train
python scripts/precompute_features.py --config configs/train_colab_mae.yaml --split val
python scripts/train_cached.py --config configs/train_colab_mae.yaml --no-amp
python scripts/eval.py        --config configs/train_colab_mae.yaml

# LoRA (Faz 2) — cache YOK, normal train.py
python scripts/train.py --config configs/train_colab_mae_lora.yaml

# analiz
python scripts/benchmark_latency.py --config configs/train_colab_mae.yaml --batch 1
python scripts/visualize.py --config ... --checkpoint ... --image-ids 285 724 802
python scripts/infer_video.py --config ... --checkpoint ... --video demo.mp4 --out out.mp4
```

---

## Sık karşılaşılan sorunlar

**`pycocotools` Windows'ta derlenmiyor**
C++ derleyicisi ister. Çözüm: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(*Desktop development with C++*) kur, ya da conda kullan: `conda install -c conda-forge pycocotools`.

**`torch.cuda.is_available()` False**
Sürücün CUDA 12.1'den eskiyse `requirements-gpu.txt` içindeki URL'de `cu121` → `cu118` yap ve
torch'u yeniden kur. `nvidia-smi` çıktısındaki "CUDA Version" sürücünün desteklediği üst sınırdır.

**`ModuleNotFoundError: mtl`**
`pip install -e .` çalıştırılmamış (src-layout). Sanal ortamın etkin olduğundan emin ol.

**HF ağırlıkları inmiyor (BEiT/I-JEPA/MoCo)**
```bash
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com     # ayna
export MTL_WEIGHTS_DIR=/yol/weights          # elle indirilen ağırlıklar burada aranır
```

**CUDA OOM**
Config'te `train.batch_size` düşür (4 → 2). LoRA koşuları donuk rejimden belirgin daha çok VRAM ister.
