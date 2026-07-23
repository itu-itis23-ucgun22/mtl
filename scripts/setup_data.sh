#!/usr/bin/env bash
# Veri hazırlama (yeni makinede). Repo kökünden:  bash scripts/setup_data.sh
#
# ÖN KOŞUL: annotation JSON'ları elde olmalı ->  data/coco_subset/annotations/
#     instances_train_subset.json · instances_val_subset.json · [instances_test_subset.json]
# Bunlar KÜÇÜK (birkaç MB) — Drive'dan/USB'den kopyala. 8 GB görüntü taşımana GEREK YOK:
# bu script görüntüleri JSON'lardaki id'lere göre COCO sunucusundan indirir.
#
# ⭐ NEDEN BÖYLE: JSON'lar alt-kümeyi birebir sabitler -> yeni makinede AYNI veri seti olur,
#    yani LoRA/Faz-2 sonuçları donuk baseline ile kıyaslanabilir kalır (delta geçerli).
set -euo pipefail
cd "$(dirname "$0")/.."

ANN="data/coco_subset/annotations"
IMG="data/coco_subset/images"
PY="${PYTHON:-python}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

echo "==> 1/3  Annotation kontrolü ($ANN)"
missing=0
for f in instances_train_subset.json instances_val_subset.json; do
    if [ ! -f "$ANN/$f" ]; then echo "    EKSİK: $ANN/$f"; missing=1; else
        echo "    var: $f  ($(du -h "$ANN/$f" | cut -f1))"; fi
done
if [ "$missing" = "1" ]; then
    cat <<'EOF'

HATA: annotation JSON'ları yok. Bunları eski makineden/Drive'dan kopyala:
    data/coco_subset/annotations/instances_train_subset.json
    data/coco_subset/annotations/instances_val_subset.json
    data/coco_subset/annotations/instances_test_subset.json   (varsa)
Küçük dosyalar — USB / Drive indirme / scp hepsi olur.
EOF
    exit 1
fi

echo "==> 2/3  Görüntüleri indir (COCO sunucusundan, paralel)"
mkdir -p "$IMG/train" "$IMG/val"
$PY scripts/download_subset_images.py --ann-file "$ANN/instances_train_subset.json" \
    --out-dir "$IMG/train" --workers 16
$PY scripts/download_subset_images.py --ann-file "$ANN/instances_val_subset.json" \
    --out-dir "$IMG/val" --workers 16
# test alt-kümesi de val2017 görüntülerinden gelir -> aynı klasöre iner
if [ -f "$ANN/instances_test_subset.json" ]; then
    $PY scripts/download_subset_images.py --ann-file "$ANN/instances_test_subset.json" \
        --out-dir "$IMG/val" --workers 16
fi

echo "==> 3/3  Doğrulama"
echo "    train: $(find "$IMG/train" -name '*.jpg' | wc -l) görsel   (beklenen ~22500)"
echo "    val  : $(find "$IMG/val"   -name '*.jpg' | wc -l) görsel   (val+test, beklenen ~4000)"
echo
echo "Veri hazır. Sonraki: python scripts/train.py --config configs/train_colab_mae_lora.yaml"
