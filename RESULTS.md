# Sonuç Karşılaştırması — ResNet vs DINO

Tüm karşılaştırma koşularının **tek toplu tablosu**. Ham makine logu: `runs/results.csv`
(Colab'da `scripts/eval.py` her eval'de satır ekler). Anlatı/gerekçe: [EXPERIMENTS.md](EXPERIMENTS.md).

Metrikler: `detection_mAP` (COCO bbox mAP), `seg_mIoU`, `cls_mAP`, `cls_F1`.

## Ana tablo

| # | backbone | trainable_layers | step | detection_mAP | seg_mIoU | cls_mAP | cls_F1 | kaynak / not |
|---|---|---|---|---|---|---|---|---|
| 1 | resnet50 | 3 | 200 | 0.0019 | 0.0208 | 0.130 | 0.232 | EXPERIMENTS Deneme 1 |
| 2 | resnet50 | 3 | 2813 | 0.0517 | 0.1218 | 0.429 | 0.409 | EXPERIMENTS Deneme 3 (1 epoch) |
| 3 | resnet50 | 3 | 4500 | 0.088 | 0.1941 | 0.5055 | 0.5035 | **sayılar bekleniyor** (sen doldur / results.csv) 
| 4 | dino_vitb16 | 0 | 200 | 0.0002 | 0.0186 | 0.0672 | 0.2510 | **sayılar bekleniyor** |
| 5 | dino_vitb16 | 0 | 2813 | 0.0095 | 0.0559 | 0.2334 | 0.2684 | **sayılar bekleniyor** |

> Deneme 2 (resnet, iddia edilen layers=0, ~200? adım): det 0.0013 / seg 0.022 / cls_mAP 0.128 /
> cls_F1 0.239 — **config doğrulanmadı, güvenilmez** (aşağıdaki uyarı), o yüzden ana tabloya
> gerçek bir "layers=0" satırı olarak alınmadı.

## Aynı adımda doğrudan kıyas

**step 200** (en erken sinyal):
| metric | resnet50 (layers=3) | dino_vitb16 (layers=0) |
|---|---|---|
| detection_mAP | 0.0019 | _?_ |
| seg_mIoU | 0.0208 | _?_ |
| cls_mAP | 0.130 | _?_ |
| cls_F1 | 0.232 | _?_ |

**step 4500**:
| metric | resnet50 (layers=3) | dino_vitb16 (layers=0) |
|---|---|---|
| detection_mAP | _?_ | _?_ |
| seg_mIoU | _?_ | _?_ |
| cls_mAP | _?_ | _?_ |
| cls_F1 | _?_ | _?_ |

## ⚠️ Eksik koşu: donuk (layers=0) ResNet hiç denenmedi

DINO **layers=0 (tamamen donuk)** koşuluyor; ama ResNet'in tüm koşuları **layers=3 (kısmen
eğitilebilir)**. Yani şu anki kıyasta backbone farkının yanında bir de **protokol farkı** var
(donuk vs kısmen eğitilebilir) → tam adil değil.

- Deneme 2 "layers=0" diye etiketli ama config'i doğrulanmadı (Deneme 3 notu da flip-flop'a
  işaret ediyor). Kullanıcı teyidi: **ResNet layers=0 hiç çalıştırılmadı.**
- **Yapılacak (adil kıyas için):** ResNet'i donuk koştur —
  `python scripts/train.py --config configs/train_colab_gpu.yaml --overrides model.trainable_backbone_layers=0`
  ve aynı step'lerde (200, 4500) değerlendir. O zaman "donuk ResNet vs donuk DINO" birebir olur.

## Nasıl güncellenir
- Colab'da her `scripts/eval.py` koşusu `runs/results.csv`'ye satır ekler.
- `python scripts/compare_results.py` → markdown tablo basar (buraya yapıştırılabilir).
- Yeni sayı geldikçe yukarıdaki `_?_` hücreleri doldurulur.
