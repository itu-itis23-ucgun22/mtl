#!/usr/bin/env bash
# Yerel kurulum (Linux / macOS). Repo kökünden çalıştır:  bash scripts/setup_local.sh
# Yaptığı: Python sürüm kontrolü -> .venv oluştur -> GPU'ya göre doğru torch'u kur
#          -> paketi editable kur -> doğrula. Kurulumdan sonra veri adımı: SETUP.md.
set -euo pipefail

cd "$(dirname "$0")/.."          # repo köküne geç
VENV=".venv"

echo "==> 1/5  Python kontrolü"
PY=$(command -v python3 || command -v python) || { echo "HATA: python bulunamadı"; exit 1; }
"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"HATA: Python >=3.10 gerekli, bulunan {sys.version.split()[0]}")
print("Python", sys.version.split()[0], "OK")
EOF

echo "==> 2/5  Sanal ortam ($VENV)"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"
PYV="$VENV/bin/python"
"$PIP" install --quiet --upgrade pip setuptools wheel

echo "==> 3/5  Bağımlılıklar"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "    NVIDIA GPU bulundu -> requirements-gpu.txt (CUDA torch)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    REQ="requirements-gpu.txt"
else
    echo "    GPU yok -> requirements-local.txt (CPU torch; eğitim çok yavaş olur)"
    REQ="requirements-local.txt"
fi
"$PIP" install -r "$REQ"

echo "==> 4/5  Paketi editable kur (src-layout: 'import mtl' için şart)"
"$PIP" install -e .

echo "==> 5/5  Doğrulama"
"$PYV" - <<'EOF'
import torch, timm, mtl
print("torch      :", torch.__version__)
print("CUDA var mı:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name(0))
print("timm       :", timm.__version__)
print("mtl paketi : OK")
EOF

cat <<'EOF'

=======================================================================
Kurulum tamam.  Ortamı etkinleştir:   source .venv/bin/activate

Hızlı test (veri gerekmez):
    pytest tests/ -q

Sonraki adım: VERİ (SETUP.md "Veri" bölümü)
    data/coco_subset/{images/{train,val},annotations/*.json}
=======================================================================
EOF
