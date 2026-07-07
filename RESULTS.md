# Sonuç Karşılaştırması — ResNet vs DINO

Tüm karşılaştırma koşularının **tek toplu tablosu**. Ham makine logu: `runs/results.csv`
(Colab'da `scripts/eval.py` her eval'de satır ekler). Anlatı/gerekçe: [EXPERIMENTS.md](EXPERIMENTS.md).

Metrikler: `detection_mAP` (COCO bbox mAP), `seg_mIoU`, `cls_mAP`, `cls_F1`.

## Ana tablo
   
| # | backbone    | trainable_layers | step | detection_mAP | seg_mIoU | cls_mAP | cls_F1 | kaynak / not |
|---|----------   |---               |------|---------------|----------|---------|--------|--------------|
| 1 | resnet50    | 3                | 200  | 0.0019        | 0.0208   | 0.130   | 0.232   | EXPERIMENTS Deneme 1 |
| 2 | resnet50    | 3                | 2813 | 0.0517        | 0.1218   | 0.429   | 0.409   | EXPERIMENTS Deneme 3 (1 epoch) |
| 3 | resnet50    | 3                | 4500 | 0.088         | 0.1941   | 0.5055  | 0.5035 | ,
| 4 | resnet50    | 0                | 2813 | 0.0467        | 0.0780   | 0.4128  | 0.4562 | donuk ResNet (EXPERIMENTS Deneme 4) |
| 5 | dino_vitb16 | 0                | 200  | 0.0002        | 0.0186   | 0.0672  | 0.2510 | 
| 6 | dino_vitb16 | 0                | 2813 | 0.0095        | 0.0559   | 0.2334  | 0.2684 | adil kıyaslama olmadığını fark ettim. 5625 adım olması gerekiyor 1 epoch'un tamamlanabilmesi için.
| 7 | dino_vitb16 | 0                | ~90000 | 0.1541      | 0.3928   | 0.5565  | 0.5515 | **16 epoch tam koşu** — EXPERIMENTS Deneme 5 |
| 8 | resnet50    | 0                | ~90000 | 0.1965      | 0.3215   | 0.7084  | 0.6799 | **16 epoch donuk — ADİL KIYAS (Deneme 6)**; batch 4. (results.csv layers=3 yanlış logladı; doğrusu 0) |


> Deneme 2 (resnet, iddia edilen layers=0, ~200? adım): det 0.0013 / seg 0.022 / cls_mAP 0.128 /
> cls_F1 0.239 — **config doğrulanmadı, güvenilmez**, o yüzden ana tabloya gerçek bir "layers=0"
> satırı olarak alınmadı. Gerçek donuk-ResNet koşusu artık satır 6'dır.


## Adil kıyas: donuk ResNet vs donuk DINO (ikisi de layers=0, step 2813)

Artık her iki omurga da **aynı protokolde** (tamamen donuk backbone) ve **aynı adımda** (2813 =
1 epoch) değerlendirildi — backbone farkı dışında değişken yok, birebir adil.

| metric | resnet50 (layers=0) | dino_vitb16 (layers=0) | kazanan |
|---|---|---|---|
| detection_mAP | **0.0467** | 0.0095 | ResNet (~4.9x) |
| seg_mIoU | **0.0780** | 0.0559 | ResNet |
| cls_mAP | **0.4128** | 0.2334 | ResNet |
| cls_F1 | **0.4562** | 0.2684 | ResNet |

**Sonuç:** donuk-vs-donuk adil kıyasta ResNet50+FPN dört metrikte de DINO ViT-B/16 + Simple
Feature Pyramid'i açık ara geçti. Beklentinin (DINO cls/seg'de öne çıkar) aksine, bu rejimde
(~22.5k subset, 1 epoch, donuk backbone) ResNet önde. Olası nedenler: (a) FPN'in gerçek çok-ölçekli
hiyerarşisi vs Simple Feature Pyramid'in tek stride-16 haritadan türetmesi — özellikle detection/seg
spatial görevlerinde; (b) DINO ViT donuk haldeyken bu head'lere/neck'e adaptasyon için daha çok
adım isteyebilir. Not: DINO'nun `layers>0` (kısmen çözülmüş) veya daha uzun koşuyla açığı kapatıp
kapatmadığı ayrı bir deney.

## DINO tam koşu (16 epoch) — büyük sıçrama, AMA henüz adil kıyas değil

DINO donuk (layers=0) **16 epoch** (~90000 adım) koşuldu. Önceki yarım-epoch DINO'ya göre
dört metrik de dramatik iyileşti:

| metric | DINO 0.5 epoch (satır 6, step 2813) | **DINO 16 epoch (satır 7)** | kazanç |
|---|---|---|---|
| detection_mAP | 0.0095 | **0.1541** | ~16x |
| seg_mIoU | 0.0559 | **0.3928** | ~7x |
| cls_mAP | 0.2334 | **0.5565** | ~2.4x |
| cls_F1 | 0.2684 | **0.5515** | ~2x |

**Yorum:** donuk DINO mimarisi (Simple Feature Pyramid + head'ler) yeterli eğitimle sağlam
öğreniyor — "donuk DINO zayıf" değil, "az eğitilmişti". detection 0.1541 mutlak olarak düşük
görünse de bu kısıt seti için (donuk backbone + sıfırdan neck/head + 22.5k subset + LR scheduler
yok) makul; COCO çıktısında IoU=0.50'de mAP **0.301**.

> ✅ **Adil kıyas artık yapıldı** (donuk ResNet 16 epoch, batch 4 — satır 8). Sonuç aşağıdaki bölümde.

## 🎯 BULGU — Adil kıyas: donuk ResNet vs donuk DINO (16 epoch, batch 4, tek değişken backbone)

Projenin ilk **confound'suz** karşılaştırması. Her şey birebir eşit — `trainable_backbone_layers=0`
(ikisi de donuk), `epochs=16`, `batch_size=4` (→ ikisi de ~90000 adım), aynı lr/wd/img_size/seed/
loss-ağırlıkları/augmentation. **Tek fark: backbone.** Bu yüzden farklar doğrudan backbone'a atfedilebilir.

| metrik | resnet50 (layers=0) | dino_vitb16 (layers=0) | kazanan |
|---|---|---|---|
| detection_mAP | **0.1965** | 0.1541 | ResNet (+%27) |
| seg_mIoU | 0.3215 | **0.3928** | **DINO (+%22)** |
| cls_mAP | **0.7084** | 0.5565 | ResNet (+%27) |
| cls_F1 | **0.6799** | 0.5515 | ResNet (+%23) |

**Bulgu: "biri diğerini ezdi" değil — görev tipine göre backbone tercihi değişir.**
- **Supervised ResNet → detection + classification'da önde.** ImageNet supervised pretraining zaten
  bir sınıflandırma görevi; image-cls'de çok güçlü (0.71), nesne-bazlı detection'a da iyi transfer.
- **Self-supervised DINO → segmentation'da önde.** Literatürle tutarlı: DINO feature'ları/attention'ı
  nesneleri dense/spatial ayırmakta güçlü (DINO'nun bilinen özelliği). SSL feature'ların uzamsal-
  semantik yapısı piksel-bazlı görevde parlıyor.

**Kısaca:** tanıma/tespit için supervised omurga, dense segmentasyon için SSL omurga. Az-etiketli
(~22.5k) çok-görevli dense tahmin rejiminde savunulabilir, nüanslı bir sonuç.

> Not (metodoloji): Deneme 4'teki eski "donuk kıyas" (step 2813) her metrikte ResNet'i gösteriyordu
> ama o **yarım epoch DINO** (batch confound) + 1 epoch'tu. 16 epoch'a çıkınca DINO segmentation'da
> öne geçti → kısa koşulardan erken sonuç çıkarmanın tehlikesinin somut kanıtı.

## Nasıl güncellenir
- Colab'da her `scripts/eval.py` koşusu `runs/results.csv`'ye satır ekler.
- `python scripts/compare_results.py` → markdown tablo basar (buraya yapıştırılabilir).
- Yeni sayı geldikçe yukarıdaki `_?_` hücreleri doldurulur.
