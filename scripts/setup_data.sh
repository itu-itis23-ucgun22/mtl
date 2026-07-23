#!/usr/bin/env bash
# Veriyi SIFIRDAN hazırla (yeni makinede, Drive/USB GEREKMEZ). Repo kökünden:
#     bash scripts/setup_data.sh
#
# ⭐ NEDEN GÜVENLİ: prepare_coco_subset.py seed'li (varsayılan 42) ve kaynak COCO annotation'ları
#    değişmez -> aşağıdaki komutlar orijinal alt-kümenin BİREBİR AYNISINI üretir. Böylece yeni
#    makinedeki sonuçlar (LoRA/Faz-2) mevcut donuk baseline ile kıyaslanabilir kalır.
#    Komutlar README + notebook'taki orijinallerle aynı (hepsi varsayılan parametre, sadece --n-images).
#
# Süre: annotations ~250 MB indirme + ~24.500 görüntü (~4 GB). 100 Mbit'te toplam ~30-45 dk.
set -euo pipefail
cd "$(dirname "$0")/.."

ANN="data/coco_subset/annotations"
IMG="data/coco_subset/images"
RAW="data/coco_raw"
PY="${PYTHON:-python}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
mkdir -p "$ANN" "$IMG" "$RAW"

echo "==> 1/4  COCO annotation'ları (kaynak)"
if [ ! -f "$RAW/annotations/instances_train2017.json" ]; then
    echo "    indiriliyor: annotations_trainval2017.zip (~250 MB)"
    # -4: IPv6 takılmasını önler · https: bazı ağlar düz http'yi bloklar · retry: geçici kopmalar
    ( cd "$RAW" && curl -4 -L --retry 5 --retry-delay 3 --connect-timeout 20 \
        -O https://images.cocodataset.org/annotations/annotations_trainval2017.zip \
      && unzip -q -o annotations_trainval2017.zip && rm -f annotations_trainval2017.zip )
else
    echo "    zaten var, atlanıyor"
fi

echo "==> 2/4  Alt-küme JSON'ları (seed 42, varsayılan parametreler)"
if [ ! -f "$ANN/instances_train_subset.json" ]; then
    $PY scripts/prepare_coco_subset.py --ann-file "$RAW/annotations/instances_train2017.json" \
        --out "$ANN/instances_train_subset.json" --n-images 22500
fi
if [ ! -f "$ANN/instances_val_subset.json" ]; then
    $PY scripts/prepare_coco_subset.py --ann-file "$RAW/annotations/instances_val2017.json" \
        --out "$ANN/instances_val_subset.json" --n-images 2000
fi
# test: val ile AYRIK olsun diye val subset'i hariç tutulur
if [ ! -f "$ANN/instances_test_subset.json" ]; then
    $PY scripts/prepare_coco_subset.py --ann-file "$RAW/annotations/instances_val2017.json" \
        --out "$ANN/instances_test_subset.json" --n-images 2000 \
        --exclude "$ANN/instances_val_subset.json"
fi

echo "==> 3/4  Görüntüler (yalnız alt-kümenin referans ettikleri; mevcutları atlar)"
$PY scripts/download_subset_images.py --ann-file "$ANN/instances_train_subset.json" \
    --out-dir "$IMG/train" --workers 16
$PY scripts/download_subset_images.py --ann-file "$ANN/instances_val_subset.json" \
    --out-dir "$IMG/val"   --workers 16
$PY scripts/download_subset_images.py --ann-file "$ANN/instances_test_subset.json" \
    --out-dir "$IMG/test"  --workers 16

echo "==> 4/4  Doğrulama"
echo "    train: $(find "$IMG/train" -name '*.jpg' | wc -l)  (beklenen 22500)"
echo "    val  : $(find "$IMG/val"   -name '*.jpg' | wc -l)  (beklenen 2000)"
echo "    test : $(find "$IMG/test"  -name '*.jpg' | wc -l)  (beklenen 2000)"
$PY - <<'EOF'
import json
v = {i['id'] for i in json.load(open('data/coco_subset/annotations/instances_val_subset.json'))['images']}
t = {i['id'] for i in json.load(open('data/coco_subset/annotations/instances_test_subset.json'))['images']}
print(f"    val={len(v)}  test={len(t)}  ÇAKIŞMA={len(v & t)}   (0 olmalı)")
EOF

echo
echo "Veri hazır. Sonraki: $PY scripts/train.py --config configs/train_colab_mae_lora.yaml"
