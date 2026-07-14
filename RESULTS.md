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
| 9 | dinov2_vitb14_reg | 0          | ~90000 | 0.2300      | 0.6011   | 0.7800  | 0.7239 | **16 epoch donuk — DÖRT METRİKTE DE EN İYİ (Deneme 7)**; batch 4, img 518 (patch14) |
| 10 | clip_vitb16 | 0                | ~90000 | 0.1420      | 0.4398   | 0.6899  | 0.6501 | **16 epoch donuk — CLIP (Deneme 8)**; batch 4, img 512 (patch16, DINOv1 ile AYNI grid); cache'li, fp32 (--no-amp). AP@0.50=0.305 |
| 11 | mae_vitb16  | 0                | ~90000 | 0.1336      | 0.2545   | 0.4215  | 0.4181 | **16 epoch donuk — MAE (Deneme 9)**; batch 4, img 512 (patch16, aynı grid). **DÖRT METRİKTE DE SON** — donuk MAE zayıf (beklenen; bkz. bulgu). AP@0.50=0.241, small AP **0.048 (ViT'lerin en yükseği)** |


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

## 🎯🎯 GÜNCEL BULGU — Üç omurga: ResNet vs DINOv1 vs DINOv2 (hepsi donuk, 16 epoch, batch 4)

DINOv2 eklendi (Deneme 7). Üçü de aynı protokolde (donuk, 16 epoch, batch 4, aynı lr/seed/loss/aug),
tek fark backbone → tam kontrollü.

| metrik | resnet50 (supervised) | dino_vitb16 (SSL v1) | **dinov2_vitb14_reg (SSL v2)** | kazanan |
|---|---|---|---|---|
| detection_mAP | 0.1965 | 0.1541 | **0.2300** | **DINOv2** |
| seg_mIoU | 0.3215 | 0.3928 | **0.6011** | **DINOv2** (açık ara) |
| cls_mAP | 0.7084 | 0.5565 | **0.7800** | **DINOv2** |
| cls_F1 | 0.6799 | 0.5515 | **0.7239** | **DINOv2** |

**Bulgu: DINOv2 dört metrikte de en iyi — ve segmentasyonda uçtan (0.60 vs ResNet 0.32 / DINOv1 0.39).**
Bu, önceki iki-yönlü bulguyu **rafine ediyor:** DINOv1 vs ResNet'te "görev tipine göre değişir" (ResNet
det/cls, DINO seg) demiştik. Ama **daha güçlü SSL (DINOv2)** gelince tablo değişiyor — SSL omurga artık
**detection ve classification'da bile** supervised ResNet'i geçiyor, sadece seg'de değil. Yani "supervised
tanıma/tespitte önde" sonucu **DINOv1'e özgüydü**; pretraining kalitesi artınca SSL her yerde öne çıkıyor.

### ⚠️ Yorum nüansı (WHY): DINOv2'nin üstünlüğü üç şeyin toplamı
DINOv2'nin farkı sadece "daha iyi SSL yöntemi/veri (LVD-142M)" değil; **kontrol edilmeyen iki ek fark** var:
1. **patch14 + img 518** → feature grid **37×37** (DINOv1 patch16@512 → 32×32; ResNet-FPN farklı).
   Daha ince grid, özellikle **dense segmentasyon**'da avantaj sağlar — 0.60 mIoU'nun bir kısmı bundan olabilir.
2. **register token'lar** (`_reg`) → daha temiz attention.

Bunlar "DINOv2'nin parçası" (haksız tweak değil), ama üstünlüğün ne kadarı **pretraining kalitesi** ne kadarı
**ince patch/register** ondan emin olmak için: DINOv2'yi patch16-eş bir ayarla ya da `_reg`'siz koşmak ayrı
bir ablasyon olur. Yine de mutlak sonuç net: **donuk DINOv2, bu multi-task dense rejimde en güçlü omurga.**
(Küçük ilginçlik: ince patch'e rağmen DINOv2 small-object AP hâlâ düşük, 0.036 — büyük/orta nesnede çok güçlü.)

## 🎯🎯🎯 GÜNCEL BULGU — Dört omurga + EN TEMİZ kıyas: CLIP vs DINOv1 (donuk, 16 epoch, batch 4)

CLIP ViT-B/16 eklendi (Deneme 8). Dördü de aynı protokolde. CLIP, **DINOv1 ile aynı ViT-B/16 + patch16
+ 32×32 grid** olduğu için aralarındaki **tek fark pretraining sinyali** (dil-contrastive vs SSL) → tezin
"pretraining sinyali downstream'i öngörür mü?" sorusunun en confound'suz test noktası (DINOv2'deki patch/register
karışıklığı yok).

| metrik | resnet50 (supervised) | dino_vitb16 (SSL) | **clip_vitb16 (dil)** | dinov2_vitb14_reg (SSL v2) | sıralama |
|---|---|---|---|---|---|
| detection_mAP | 0.1965 | 0.1541 | **0.1420** | **0.2300** | DINOv2 > ResNet > DINOv1 > **CLIP** |
| seg_mIoU | 0.3215 | 0.3928 | **0.4398** | **0.6011** | DINOv2 > **CLIP** > DINOv1 > ResNet |
| cls_mAP | 0.7084 | 0.5565 | **0.6899** | **0.7800** | DINOv2 > ResNet > **CLIP** > DINOv1 |
| cls_F1 | 0.6799 | 0.5515 | **0.6501** | **0.7239** | DINOv2 > ResNet > **CLIP** > DINOv1 |

**Bulgu 1 — CLIP vs DINOv1 (aynı mimari, farklı pretraining):** CLIP, DINOv1'i **seg + iki cls metriğinde
açık ara** geçiyor (cls_mAP 0.69 vs 0.56), sadece **detection'da hafif geride** (0.142 vs 0.154). Yani
**dil-contrastive pretraining, SSL-distillation'a göre daha güçlü SEMANTİK feature** veriyor — image-text
eşleştirme amacı doğrudan tanımaya (classification) ve nesne-semantiğine (segmentation) hizmet ediyor.

**Bulgu 2 — CLIP'in zayıf noktası: LOKALİZASYON.** CLIP detection'da **dört omurganın en düşüğü** (0.142),
AP@0.75 sadece 0.118. Bu da beklenen: CLIP'in image-text kontrastif amacı **global/semantik**tir, kesin
kutu lokalizasyonunu ödüllendirmez. "İyi tanır, kötü yerleştirir" → dense tahminde net bir karakter.

**Bulgu 3 — genel resim:** DINOv2 hâlâ **dört metrikte de lider**. Omurgalar bir spektrum çiziyor:
supervised (ResNet) tanıma/tespit dengeli; SSL-v1 (DINOv1) seg-eğilimli ama zayıf; **dil (CLIP)
semantik-güçlü/lokalizasyon-zayıf**; SSL-v2 (DINOv2) her yerde en iyi. Tezin cevabı **evet**: pretraining
sinyali, hangi downstream görevde parlayacağını öngörüyor.

> Not (küçük confound): CLIP koşusu fp32 (`--no-amp`, focal-loss NaN'ından kaçınmak için), diğerleri AMP'liydi.
> fp32 sayısal olarak daha doğru olduğundan bu CLIP'i haksız yere zayıflatmaz; nihai metrikleri anlamlı etkilemez.
> Ayrıca CLIP kendi native normalizasyonuyla beslendi (clip_backbone.py içinde ImageNet→CLIP), her omurga
> kendi ön-işlemesini aldığı için adil.

## 🎯🎯🎯🎯 BULGU — MAE: "donukken kötü, çözüldüğünde iyi" (Deneme 9)

MAE ViT-B/16 eklendi — sweep'in **sıfır confound'lu** üyesi (DINOv1/CLIP/SAM ile aynı ViT-B, patch16,
512, 32×32 grid, ImageNet norm, SFP, batch 4, 16 epoch). **Dört metrikte de SON.**

| metrik | DINOv2 | ResNet | CLIP | DINOv1 | **MAE** |
|---|---|---|---|---|---|
| detection_mAP | **0.2300** | 0.1965 | 0.1420 | 0.1541 | **0.1336** |
| seg_mIoU | **0.6011** | 0.3215 | 0.4398 | 0.3928 | **0.2545** |
| cls_mAP | **0.7800** | 0.7084 | 0.6899 | 0.5565 | **0.4215** |
| cls_F1 | **0.7239** | 0.6799 | 0.6501 | 0.5515 | **0.4181** |

**Bulgu: bu bir başarısızlık değil, MAE'nin BİLİNEN imzası.** MAE **pikselleri yeniden kurmayı**
öğrenir; piksel kurmak **düşük seviyeli doku/detay** ister, **semantik soyutlama gerektirmez**
("bu bir köpek" bilgisi olmadan da pikseller kurulabilir). Sonuç: donuk MAE feature'ları semantik
olarak **organize değil** → linear-probe / donuk transferde zayıf. MAE makalesi bunu kendisi raporlar:
**ImageNet linear probe %68 (zayıf) ama fine-tune %83.6 (DINO'yu geçer, SOTA)**. Yani MAE'nin karakteri:
> **"donukken kötü, çözüldüğünde harika."**

**Destekleyici kanıt (bizim çıktımızda):** MAE'nin **small-object AP'si 0.048** — ViT'lerin **en yükseği**
(CLIP 0.031, DINOv2 0.036). Yani **düşük seviyeli/yerel detay korunmuş**; eksik olan **semantik**. Teori
ve ölçüm birebir tutuyor.

### 🔮 Test edilebilir tahmin (ROADMAP Faz 2)
Bu bulgu, adaptasyon ekseni için **net bir öngörü** üretiyor:
> **Backbone çözüldüğünde (LoRA / full fine-tune) EN BÜYÜK kazancı MAE almalı**, DINOv2 ise en azını
> (zaten donukken tavana yakın). Sıralama fine-tune'da **değişmeli**.

Bu doğrulanırsa tez çok güçlenir: *pretraining sinyali sadece "hangi görevde iyi"yi değil,
"hangi ADAPTASYON REJİMİNDE iyi"yi de öngörüyor.*

### ⚠️ Kapsam notu (dürüstlük)
Protokolümüz **donuk**. Yani ölçtüğümüz şey **"donuk feature kalitesi"** — MAE'yi sistematik olarak
dezavantajlı kılar. Bu **haksızlık değil** (protokol herkese eşit) ama **sonucun kapsamı sınırlı**:
"MAE bu rejimde zayıf" diyebiliriz, **"MAE kötü bir backbone"** diyemeyiz. Faz 2 (adaptasyon) bunu
kapatmak için var.

## Nasıl güncellenir
- Colab'da her `scripts/eval.py` koşusu `runs/results.csv`'ye satır ekler.
- `python scripts/compare_results.py` → markdown tablo basar (buraya yapıştırılabilir).
- Yeni sayı geldikçe yukarıdaki `_?_` hücreleri doldurulur.
