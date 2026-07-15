# Deney Günlüğü

Bu dosya kod veya mimari değil, **ne denedik, ne çıktı, neden** kaydı.
Sistemin nasıl çalıştığı için [ARCHITECTURE.md](ARCHITECTURE.md)'ye bakın.

---

## Deneme 1 — 2026-07-02 — İlk gerçek COCO sanity run (backbone kısmen eğitilebilir)

### Kurulum
- `trainable_backbone_layers: 3` (o zamanki default) — ResNet50 gövdesinin son
  3 katmanı (layer2-4) eğitilebilir, stem+layer1 donuk.
- İlk kez sentetik/küçük örnek değil, gerçek ~22.500 görüntülük COCO subset
  ile çalıştırıldı (Colab'da `prepare_coco_subset.py` + `download_subset_images.py`).
- `--overrides train.max_steps=200` — tam eğitim değil, kısa bir "pipeline
  gerçek veriyle uçtan uca çalışıyor mu" doğrulaması.
- `batch_size=8, img_size=512, lr=0.0001, amp=true, epochs=16 (ama max_steps
  ile 200'de kesildi)`.

### Sonuç (checkpoint: `colab_gpu_epoch0.pt`, 200 adımda alındı)

| Metrik | Değer | Yorum |
|---|---|---|
| `detection_mAP` | 0.0019 | neredeyse sıfır |
| `seg_mIoU` | 0.0208 | neredeyse background-only seviyesi |
| `cls_mAP` | 0.130 | zayıf ama sıfır değil |
| `cls_F1` | 0.232 | zayıf ama sıfır değil |

### Analiz
- Loss'lar adım boyunca düzenli düştü: `total` 6.86 (step 0) → 4.95 (step 180)
  — dört görevin de (classification, bbox_regression, seg_loss, cls_loss)
  gradyanı doğru akıyor, pipeline sağlıklı.
- Görev zorluğu sıralaması ilk kez burada gözlemlendi: classification en
  hızlı öğrenen (yoğun, görüntü-başı tek etiket sinyali), detection ve
  segmentation daha yavaş (sıfırdan/piksel-bazlı, daha fazla veri/adım istiyor).
- 200 adım, toplam hedefin (~45.000 adım, `epochs=16`) sadece %0,44'ü — mAP/
  mIoU rakamlarının düşük olması **beklenen**, bir hata değil. Bu koşunun
  amacı model kalitesi değil, mimarinin gerçek veriyle uçtan uca çalıştığını
  kanıtlamaktı.

### Sonrasında yapılan değişiklik
`trainable_backbone_layers` `3`'ten `0`'a indirildi (ResNet50 gövdesi
tamamen donduruldu) → bkz. Deneme 2.

---

## Deneme 2 — 2026-07-02 — Backbone tamamen donuk

### Yapılan değişiklikler (commit `a578697`)
- `trainable_backbone_layers`: `3` → `0` — ResNet50 gövdesi tamamen donduruldu,
  sadece FPN + üç head (detection/segmentation/classification) eğitiliyor.
  (Not: `resnet_fpn_backbone`'daki `trainable_layers` sadece ResNet gövdesini
  kontrol ediyor; FPN bu parametreden bağımsız olarak her zaman eğitilebilir.)
- Mid-epoch checkpoint (`checkpoint_every_steps=500`) + `scripts/train.py --resume`
  eklendi — uzun Colab koşularının kesilmeye karşı dayanıklı olması için.

### Karşılaşılan sorun: Colab'da "step 0'da takılma"
Eğitim uzun süre step 0'da ilerlemiyor gibi görünmüştü. Olası nedenler tartışıldı
(DataLoader `num_workers=2` + Colab'ın küçük `/dev/shm`'i, GPU runtime seçili
olmama ihtimali, pretrained ağırlık indirme). Kesin kök neden bu oturumda
doğrulanmadı — bir sonraki koşuda `Config:` ve ilk `step=0` log satırının ne
zaman göründüğü izlenerek ayırt edilecek.

### Sonuç (eval çıktısı, checkpoint'in kaç adımda alındığı doğrulanmadı)

| Metrik | Değer | Yorum |
|---|---|---|
| `detection_mAP` | 0.0013 | neredeyse sıfır |
| `seg_mIoU` | 0.022 | neredeyse background-only seviyesi |
| `cls_mAP` | 0.128 | zayıf ama sıfır değil |
| `cls_F1` | 0.239 | zayıf ama sıfır değil |

### Analiz
- Kod tarafında yapısal bir engel **yok**: backbone donuk olsa da gradyan
  head'lere tam akıyor, `AdamW(model.parameters())` tüm head parametrelerini
  kapsıyor ([multitask_model.py](src/mtl/models/multitask_model.py),
  [train.py:73](scripts/train.py#L73)). Doğrulandı.
- Detection head ([detection_head.py](src/mtl/models/detection_head.py))
  tamamen sıfırdan (RetinaNet'in kendi head'i pretrained değil) — bu yüzden
  en yavaş öğrenen görev.
- Görev zorluğu sıralaması sonuçlarla tutarlı: classification (en az yeni
  parametre, frozen ImageNet feature'larla en uyumlu) > segmentation >
  detection (sıfırdan + spatial).
- **En olası açıklama:** checkpoint yeterince az adımda alınmış (tam
  `epochs=16` / ~45.000 adımın tamamı değil), yani bir bug değil, azlık.

### Sonraki adım (henüz çalıştırılmadı)
Tam koşu (`max_steps: null`, zaten default) ~45.000 adım, T4 + AMP ile tahmini
**4-7 saat**. Bu, tek seferde beklemek için uzun bulundu. Kararlaştırılan plan:
1. Full run'ı arka planda başlatıp ara checkpoint'leri (`_step3000.pt`,
   `_step6000.pt`, ...) periyodik olarak `scripts/eval.py` ile değerlendirerek
   trend'i izlemek (loss düşüyor mu, mAP artıyor mu), tam bitmesini beklemeden.
2. Alternatif: çok daha küçük bir "sinyal" koşusu (`n_images=3000, epochs=3`,
   ~1.125 adım, ~10-15 dk) ile mimarinin genel olarak öğrenip öğrenmediğini
   hızlıca doğrulamak. Henüz bir config dosyası oluşturulmadı.

---

## Deneme 3 — 2026-07-02 — Tam 1 epoch (2813 adım)

### Kurulum
- `--overrides train.max_steps=2813` — tam 1 epoch'a denk gelen adım sayısı
  (`ceil(22500 / 8) = 2813`), 200 adımlık kısa doğrulamadan sonraki doğal
  ara adım.
- **Not (doğrulanmadı, config dosyasının şu anki haline göre çıkarım):**
  bu koşu sırasında `configs/train_colab_gpu.yaml`'da
  `trainable_backbone_layers: 3` yazıyordu (Deneme 2'de 0'a çekilmişti,
  sonra tekrar 3'e alınmış görünüyor) — yani bu koşu muhtemelen Deneme 1
  ile aynı backbone ayarını kullandı, Deneme 2'nin donuk-backbone ayarını
  değil. Doğrulanırsa bu not güncellenmeli.
- Diğer tüm ayarlar Deneme 1/2 ile aynı: `batch_size=8, img_size=512,
  lr=0.0001, amp=true`, gerçek ~22.500 görüntülük COCO subset.

### Sonuç

| Metrik | Deneme 1/2 (200 adım) | Deneme 3 (2813 adım = 1 epoch) | Değişim |
|---|---|---|---|
| `detection_mAP` | ~0.0013-0.0019 | **0.0517** | ~30x |
| `seg_mIoU` | ~0.021-0.022 | **0.1218** | ~6x |
| `cls_mAP` | ~0.128-0.130 | **0.429** | ~3.3x |
| `cls_F1` | ~0.232-0.239 | **0.409** | ~1.8x |

### Analiz
- Dört metrik de ~14 kat daha fazla adımda tutarlı ve belirgin şekilde
  iyileşti — rastgele gürültü değil, gerçek bir öğrenme trendi. Deneme 1/2'deki
  çok düşük sayıların bir bug değil, sadece "yetersiz eğitim" olduğunun en
  net kanıtı.
- Referans için: tam eğitilmiş bir RetinaNet-R50-FPN COCO'da ~%35-37 mAP'a
  ulaşır. 1 epoch'ta (planın 1/16'sı) %5,2'deyiz — uzak ama makul bir
  yörünge.
- Görev zorluğu sıralaması burada da doğrulandı: classification en önde
  (%42,9 mAP), detection en gerideki (%5,2) — sıfırdan/en karmaşık görev.

### Sonraki adım
Trend sağlıklı olduğu için uzun/tam koşuya (`epochs=16`, ~45.000 adım) devam
etmeye karar verildi. `checkpoint_every_steps=500` + `scripts/train.py
--resume` sayesinde Colab oturum kesintilerine karşı güvenli.

---

## Deneme 4 — 2026-07-04 — Donuk (layers=0) ResNet, 1 epoch — eksik adil-kıyas koşusu

### Kurulum
- `--overrides model.trainable_backbone_layers=0 train.max_steps=2813` — ResNet50 gövdesi
  **tamamen donuk**, sadece FPN + üç head eğitiliyor. Tam 1 epoch (2813 adım).
- Amaç: DINO'nun tüm koşuları `layers=0` (donuk), ResNet'inkiler ise `layers=3` (kısmen
  eğitilebilir) olduğu için kıyasta bir **protokol farkı** vardı (bkz. aşağıdaki düzeltme).
  Bu koşu o eksiği kapatıyor — artık "donuk ResNet vs donuk DINO", aynı adımda birebir.
- Diğer ayarlar Deneme 1–3 ile aynı: `batch_size=8, img_size=512, lr=0.0001, amp=true`,
  gerçek ~22.500 görüntülük COCO subset. Checkpoint: `colab_gpu_epoch0.pt`.

### Sonuç (donuk ResNet, 2813 adım)

| Metrik | Donuk ResNet (Deneme 4) | Donuk DINO (2813, aynı protokol) | Kısmen-eğitilebilir ResNet (Deneme 3) |
|---|---|---|---|
| `detection_mAP` | 0.0467 | 0.0095 | 0.0517 |
| `seg_mIoU` | 0.0780 | 0.0559 | 0.1218 |
| `cls_mAP` | 0.4128 | 0.2334 | 0.429 |
| `cls_F1` | 0.4562 | 0.2684 | 0.409 |

### Analiz
- **Adil kıyas (donuk vs donuk, step 2813): ResNet dört metrikte de DINO'yu geçti** —
  detection'da ~4.9x, classification'da ~1.8x. Beklentinin (DINO cls/seg'de öne çıkar)
  aksine bu rejimde (az veri, 1 epoch, donuk) ResNet feature'ları + gerçek FPN daha güçlü.
- ResNet donuk (0.0467) vs kısmen-eğitilebilir (0.0517) detection farkı küçük → 1 epoch'ta
  backbone'u çözmenin katkısı henüz sınırlı; asıl fark omurga seçiminde (ResNet vs DINO),
  donuk/çözük ayarında değil. Segmentation'da ise çözük backbone belirgin fark yaratıyor
  (0.078 → 0.122), yani spatial görev backbone adaptasyonundan daha çok faydalanıyor.
- Tam toplu tablo ve yorum: [RESULTS.md](RESULTS.md) (satır 6 + "Adil kıyas" bölümü).

### Sonraki adım (opsiyonel)
DINO'nun açığı protokolden mi yoksa omurgadan mı geldiğini ayırmak için: DINO `layers>0`
(kısmen çözük) veya daha uzun koşu. Ama mevcut adil kıyas net bir sinyal veriyor.

---

## Deneme 5 — 2026-07-07 — DINO donuk, TAM 16 epoch (~90.000 adım)

### Kurulum
- `configs/train_colab_dino.yaml` tam koşu: `backbone=dino_vitb16, trainable_backbone_layers=0`
  (donuk), `batch_size=4, epochs=16` → 16 × 5625 = ~90.000 adım. Checkpoint: `colab_dino_epoch15.pt`.
- Colab oturum kesintileri + veri (Drive throttling) sorunları nedeniyle koşu birkaç oturuma
  bölündü; `checkpoint_every_steps=500` + `scripts/train.py --resume` (global adımdan devam eden
  yama, commit `e7164c6`) ile kaldığı yerden tamamlandı. Veri bütünlüğü doğrulandı (eksik görüntü
  olsa `Image.open` çökerdi; koşu çökmeden bitti → 22.500 görüntü tamdı).

### Sonuç

| Metrik | DINO 0.5 epoch (Deneme, step 2813) | **DINO 16 epoch (Deneme 5)** | Değişim |
|---|---|---|---|
| `detection_mAP` | 0.0095 | **0.1541** | ~16x |
| `seg_mIoU` | 0.0559 | **0.3928** | ~7x |
| `cls_mAP` | 0.2334 | **0.5565** | ~2.4x |
| `cls_F1` | 0.2684 | **0.5515** | ~2x |

(COCO çıktısı: IoU=0.50'de mAP 0.301; strict 0.50:0.95 = 0.1541.)

### Analiz
- Donuk DINO mimarisi (Simple Feature Pyramid + head'ler) **yeterli eğitimle sağlam öğreniyor** —
  önceki düşük sayılar "donuk DINO zayıf"tan değil, "az eğitilmiş"ten geliyormuş. Deneme 3'teki
  (ResNet) "adım arttıkça hepsi belirgin iyileşir" trendi burada DINO için de doğrulandı.
- Mutlak detection 0.1541 düşük görünse de bu kısıt seti için makul: donuk backbone + sıfırdan
  neck/head + 22.5k subset + LR scheduler yok. Ana fren **donuk backbone**; yükseltmek için en
  büyük kaldıraç epoch değil, backbone'u çözmek (fine-tune / LoRA).
- Görev sıralaması yine tutarlı: det (0.15) < seg (0.39) < cls (0.55).

### ⚠️ Kritik: bu HENÜZ bir bulgu DEĞİL (confound var)
DINO 16 epoch koştu; ama mevcut ResNet koşuları en fazla ~1.6 epoch (step 4500, üstelik layers=3).
Yani "DINO artık ResNet'i geçti" **denemez** — backbone farkının yanında **eğitim bütçesi (16 vs
1.6 epoch) ve protokol (donuk vs layers=3)** farkı da var. Tek değişkeni izole edemiyoruz.

### Sonraki adım (adil kıyası tamamla)
`configs/train_colab_resnet_frozen.yaml` oluşturuldu: **donuk ResNet (layers=0), batch_size=4,
epochs=16** — DINO ile birebir aynı (tek değişken backbone). Bu koşulunca "ikisi de donuk + 16
epoch + batch 4" gerçek adil kıyas olur → o zaman savunulabilir bir bulgu çıkar. Colab Pro alındığı
için bütçe artık yeterli. Ayrıca planlanan eksen: DINO donuk vs **LoRA** vs full fine-tune (PEFT).

---

## Deneme 6 — 2026-07-07 — 🎯 Donuk ResNet 16 epoch → İLK ADİL KIYAS (bulgu)

### Kurulum
- ResNet donuk 16 epoch: `train_colab_gpu.yaml` + `--overrides model.trainable_backbone_layers=0
  train.batch_size=4`. Böylece DINO 16-epoch koşusuyla (Deneme 5) **birebir eşit**: layers=0,
  epochs=16, batch=4 (→ ~90000 adım), aynı lr/wd/img_size/seed/loss/aug. **Tek fark: backbone.**
- **NaN olayı (step ~64000):** AMP (fp16) altında focal loss'un `log(sigmoid)` terimi, eğitilen
  detection-cls head'inin büyüyen logit'lerinde taştı → `classification=nan`. `train_one_epoch.py`
  guard'ı yakalayıp durdurdu (checkpoint temizdi). İki resume denemesi farklı shuffle'a rağmen ~aynı
  adımda tekrar NaN attı → sorun batch değil, **kırılgan ağırlıklar**. Çözüm: `--overrides
  train.amp=false` (fp32'nin geniş aralığı taşmayı önledi) ile resume → koşu tamamlandı.
- **Loglama düzeltmesi:** eval `--overrides` almadığı için results.csv'ye `trainable_layers=3`
  yazdı; doğrusu **0** (RESULTS.md satır 8'de düzeltildi).

### Sonuç — adil kıyas (ikisi de donuk, 16 epoch, batch 4)

| metrik | ResNet50 (layers=0) | DINO ViT-B/16 (layers=0) | kazanan |
|---|---|---|---|
| detection_mAP | **0.1965** | 0.1541 | ResNet (+%27) |
| seg_mIoU | 0.3215 | **0.3928** | **DINO (+%22)** |
| cls_mAP | **0.7084** | 0.5565 | ResNet (+%27) |
| cls_F1 | **0.6799** | 0.5515 | ResNet (+%23) |

### 🎯 Bulgu
Projenin ilk **confound'suz** sonucu. "Biri diğerini ezdi" değil — **görev tipine göre backbone
tercihi değişiyor:**
- **Supervised ResNet → detection + classification'da önde.** ImageNet supervised pretraining zaten
  bir sınıflandırma görevi olduğundan image-cls'de çok güçlü (0.71) ve detection'a iyi transfer.
- **Self-supervised DINO → segmentation'da önde.** DINO feature/attention'ının nesneleri dense/
  spatial ayırmadaki bilinen gücüyle tutarlı; SSL'in uzamsal-semantik yapısı piksel görevde parlıyor.

Yani: tanıma/tespit için supervised omurga, dense segmentasyon için SSL omurga.

### Metodolojik ders
Deneme 4'teki eski "donuk kıyas" (step 2813, yarım-epoch DINO + batch confound) **her** metrikte
ResNet'i gösteriyordu. 16 epoch'a çıkınca DINO segmentation'da öne geçti → **kısa koşulardan erken
sonuç çıkarmak yanıltıcı.** Confound kontrolü (eşit epoch + eşit batch + eşit protokol) bu bulguyu
mümkün kıldı.

### Sonraki adım
- **PEFT ekseni:** DINO donuk vs LoRA vs full fine-tune → backbone'u (ucuza) çözmek bulguyu nasıl
  değiştirir. LoRA 12 GB'a sığar, sayıları muhtemelen zıplatır.
- **Kalıcı sağlamlık:** grad clipping (NaN'i kökten önlemek için) — planlandı.
- İsteğe bağlı: `trainable_layers=3` ResNet'i de 16 epoch koşup "çözük omurga" ekseni.

---

## Deneme 7 — 2026-07-07 — 🎯🎯 DINOv2 donuk 16 epoch → ÜÇ OMURGA, DINOv2 DÖRT METRİKTE EN İYİ

### Kurulum
- `configs/train_colab_dinov2.yaml`: `backbone=dinov2_vitb14_reg` (register'lı ViT-B/14), donuk
  (layers=0), 16 epoch, batch 4, **img_size 518** (patch14 için 14'e bölünebilir). ResNet/DINOv1
  16-epoch koşularıyla aynı protokol (lr/wd/seed/loss/aug). Checkpoint: `colab_dinov2_epoch15.pt`.
- Koşu birkaç oturuma bölündü (runtime/Drive kesintileri); `--resume epoch13.pt` (temiz epoch
  sınırı) ile tamamlandı. Not: bir ara `--overrides train.amp=false` denendi ama DINOv2 NaN
  atmadığı için gereksizdi (sadece yavaşlattı); AMP açık asıl koşu.

### Sonuç — üç omurga (hepsi donuk, 16 epoch, batch 4)

| metrik | ResNet50 | DINO ViT-B/16 (v1) | **DINOv2 ViT-B/14 (+reg)** | kazanan |
|---|---|---|---|---|
| detection_mAP | 0.1965 | 0.1541 | **0.2300** | DINOv2 |
| seg_mIoU | 0.3215 | 0.3928 | **0.6011** | DINOv2 (açık ara) |
| cls_mAP | 0.7084 | 0.5565 | **0.7800** | DINOv2 |
| cls_F1 | 0.6799 | 0.5515 | **0.7239** | DINOv2 |

(COCO çıktısı: IoU=0.50'de mAP 0.412; medium/large AP 0.29/0.42 güçlü, small 0.036 zayıf.)

### 🎯 Bulgu — önceki iki-yönlü sonucu rafine ediyor
Deneme 6'da "DINOv1 vs ResNet: göreve göre değişir (ResNet det/cls, DINO seg)" demiştik. **DINOv2
gelince tablo değişiyor:** daha güçlü SSL, **detection ve classification'da bile** supervised ResNet'i
geçiyor — sadece seg'de değil. Yani "supervised tanıma/tespitte önde" sonucu **DINOv1'e özgüymüş**;
pretraining kalitesi artınca **SSL her görevde öne çıkıyor**. Özellikle segmentasyon uçtan (0.60 vs
0.32/0.39) — SSL'in dense/spatial gücünün en net kanıtı.

### ⚠️ Yorum nüansı (kontrol edilmeyen iki fark)
DINOv2'nin üstünlüğü sadece "daha iyi SSL yöntemi/veri (LVD-142M)" değil; iki ek fark var:
1. **patch14 + img 518** → feature grid 37×37 (DINOv1 32×32'den ince) — dense seg'e avantaj.
2. **register token'lar** (`_reg`) → temiz attention.
Bunlar "DINOv2'nin parçası", haksız tweak değil; ama üstünlüğün ne kadarı **pretraining** ne kadarı
**ince patch/register** ondan emin olmak için ayrı ablasyon gerekir (patch16-eş ayar / `_reg`'siz).
Mutlak sonuç yine de net: **donuk DINOv2 bu multi-task dense rejimde en güçlü omurga.**

### Sonraki adım
- Foundation-model sweep'e devam: **CLIP** (dil-supervised) ve **SAM** (seg-native) → ROADMAP Faz 1.
  Artık feature-caching altyapısı hazır (precompute → train_cached), donuk sweep ucuz.
- PEFT ekseni (LoRA) + grad clipping hâlâ sırada.

---

## Deneme 8 — 2026-07-09 — 🎯🎯🎯 CLIP donuk 16 epoch → DÖRT OMURGA + EN TEMİZ KIYAS (CLIP vs DINOv1)

### Kurulum
- `configs/train_colab_clip.yaml`: `backbone=clip_vitb16` (OpenAI CLIP ViT-B/16), donuk (layers=0),
  16 epoch, batch 4, **img_size 512** (patch16 → 32×32 grid). Checkpoint: `colab_clip_cached_epoch15.pt`.
- **Feature-caching akışı** (ROADMAP varsayılanı): precompute trunk → train_cached (neck+head). Donuk
  trunk deterministik olduğu için ViT forward bir kez koşuldu.
- **fp32 (`--no-amp`):** cached eğitimde focal-loss fp16 NaN'ı tekrarladı; AMP kapatıldı. Cached modda
  ViT forward atlandığı için AMP faydası ~yok → fp32 bedava, NaN'ı tamamen kaldırdı (grad-clip de eklendi).
- **Normalizasyon:** CLIP kendi native norm'uyla beslendi (clip_backbone.py içinde ImageNet→CLIP;
  transforms.py'ye dokunulmadı). Her omurga kendi ön-işlemesini alır → adil.
- **Neden en temiz kıyas:** CLIP ViT-B/16, **DINOv1 ile aynı mimari + patch16 + 32×32 grid** → aralarındaki
  tek fark **pretraining sinyali** (dil-contrastive vs SSL). DINOv2'deki patch/register confound'u yok.

### Sonuç — dört omurga (hepsi donuk, 16 epoch, batch 4)

| metrik | ResNet50 | DINOv1 | **CLIP ViT-B/16** | DINOv2 | sıralama |
|---|---|---|---|---|---|
| detection_mAP | 0.1965 | 0.1541 | **0.1420** | 0.2300 | DINOv2 > ResNet > DINOv1 > **CLIP** |
| seg_mIoU | 0.3215 | 0.3928 | **0.4398** | 0.6011 | DINOv2 > **CLIP** > DINOv1 > ResNet |
| cls_mAP | 0.7084 | 0.5565 | **0.6899** | 0.7800 | DINOv2 > ResNet > **CLIP** > DINOv1 |
| cls_F1 | 0.6799 | 0.5515 | **0.6501** | 0.7239 | DINOv2 > ResNet > **CLIP** > DINOv1 |

(COCO çıktısı: AP@0.50=0.305, AP@0.75=0.118; small 0.031 / medium 0.131 / large 0.269.)

### 🎯 Bulgu
1. **CLIP vs DINOv1 (aynı mimari, farklı pretraining):** CLIP, DINOv1'i **seg + iki cls metriğinde açık ara**
   geçer (cls_mAP 0.69 vs 0.56), sadece **detection'da hafif geride** (0.142 vs 0.154). → **Dil-contrastive
   pretraining, SSL-distillation'a göre daha güçlü SEMANTİK feature** verir; image-text amacı doğrudan
   tanıma (classification) ve nesne-semantiğine (segmentation) hizmet eder.
2. **CLIP'in zayıf noktası = lokalizasyon.** Detection'da **dört omurganın en düşüğü** (0.142, AP@0.75 0.118).
   CLIP'in global/semantik amacı kesin kutu lokalizasyonunu ödüllendirmez → "iyi tanır, kötü yerleştirir".
3. **Genel resim:** DINOv2 hâlâ **dört metrikte lider**. Omurgalar bir spektrum: supervised (dengeli),
   SSL-v1 (seg-eğilimli/zayıf), **dil (semantik-güçlü/lokalizasyon-zayıf)**, SSL-v2 (her yerde en iyi).
   Tezin cevabı **evet**: pretraining sinyali hangi downstream görevde parlayacağını öngörüyor.

### Sonraki adım
- Sweep'in son paradigması: **SAM** (segmentation-native image encoder) → ROADMAP Faz 1'i kapatır.
- Sonra Faz 2 (adaptasyon: donuk → LoRA → full) ve Faz 3 ablasyonları.

---

## Deneme 9 — 2026-07-14 — 🎯 MAE donuk 16 epoch → "DONUKKEN KÖTÜ, ÇÖZÜLDÜĞÜNDE İYİ"

### Kurulum
- `configs/train_colab_mae.yaml`: `backbone=mae_vitb16` (timm `vit_base_patch16_224.mae` — **saf MAE
  pretrain**, ImageNet supervised fine-tune YOK), donuk (layers=0), 16 epoch, batch 4, img 512.
- **Sweep'in sıfır-confound'lu üyesi:** DINOv1 / CLIP / SAM ile **her eksende birebir aynı** — ViT-B
  (~86M) · patch16 · 512 · **32×32 grid** · ImageNet norm (renorm YOK) · SFP · aynı head'ler · batch 4.
  **Tek değişken: pretraining sinyali.**
- Feature-caching akışı, fp32 (`--no-amp`). Checkpoint: `colab_mae_cached_epoch15.pt`.
- **Neden MAE (I-JEPA değil):** Meta I-JEPA'nın **ViT-B'sini yayınlamadı** (sadece ViT-H/g) → boyut
  confound'u düzeltilemezdi. "Maskeli tahmin" ailesini **adil temsil edebilen tek model MAE**.

### Sonuç — beş omurga (hepsi donuk, 16 epoch, batch 4)

| metrik | DINOv2 | ResNet | CLIP | DINOv1 | **MAE** |
|---|---|---|---|---|---|
| detection_mAP | **0.2300** | 0.1965 | 0.1420 | 0.1541 | **0.1336** |
| seg_mIoU | **0.6011** | 0.3215 | 0.4398 | 0.3928 | **0.2545** |
| cls_mAP | **0.7800** | 0.7084 | 0.6899 | 0.5565 | **0.4215** |
| cls_F1 | **0.7239** | 0.6799 | 0.6501 | 0.5515 | **0.4181** |

(COCO: AP@0.50=0.241, AP@0.75=0.127; small **0.048** / medium 0.139 / large 0.230; AR@100=0.308.)

**MAE dört metrikte de SON** — ve classification'da uçurumla (0.42 vs CLIP 0.69, DINOv2 0.78).

### 🎯 Bulgu — bu bir başarısızlık değil, MAE'nin BİLİNEN imzası
MAE **maskeli pikselleri yeniden kurmayı** öğrenir. Piksel kurmak **düşük seviyeli doku/detay** ister;
**semantik soyutlama gerektirmez** — "bu bir köpek" bilgisi olmadan da pikseller kurulabilir. Dolayısıyla
donuk MAE feature'ları **semantik olarak organize değil** → donuk transferde zayıf.

Bu literatürle **birebir** uyumlu: MAE makalesi kendi raporunda **ImageNet linear probe %68 (zayıf)**
ama **fine-tune %83.6 (DINO'yu geçer)** verir. MAE'nin karakteri: **"donukken kötü, çözüldüğünde harika."**

**Destekleyici kanıt (bizim ölçümümüzde):** MAE'nin **small-object AP'si 0.048** — ViT'lerin **en yükseği**
(CLIP 0.031, DINOv2 0.036). Yani **düşük seviyeli/yerel detay korunmuş**; eksik olan **semantik**.
Teori ve ölçüm örtüşüyor → bulgunun mekanizması doğrulanmış oluyor.

### 🔮 Test edilebilir tahmin (Faz 2 için)
> Backbone çözüldüğünde (LoRA / full fine-tune) **EN BÜYÜK kazancı MAE almalı**; DINOv2 en azını
> (donukken zaten tavana yakın). **Sıralama fine-tune'da değişmeli.**

Doğrulanırsa tez ciddi güçlenir: *pretraining sinyali sadece "hangi görevde iyi"yi değil,
**"hangi adaptasyon rejiminde iyi"**yi de öngörüyor.*

### ⚠️ Kapsam notu
Protokol **donuk** → ölçtüğümüz şey **"donuk feature kalitesi"**. Bu MAE'yi sistematik dezavantaja sokar.
Haksızlık değil (protokol herkese eşit) ama sonucun kapsamı sınırlı: **"MAE bu rejimde zayıf"** denebilir,
**"MAE kötü bir backbone"** denemez. Faz 2 tam bunu kapatmak için var.

### Sonraki adım
- **SAM** koşusu → Faz 1'in son paradigması (seg-native).
- **Faz 2 (adaptasyon)**: MAE tahminini test et — çözüldüğünde en çok o kazanmalı.

---

## Deneme 10 — 2026-07-15 — 🎯 SAM donuk 16 epoch → "SEGMENTATION-NATIVE" YANILTICI (Faz 1 bitti)

### Kurulum
- `configs/train_colab_sam.yaml`: `backbone=sam_vitb16` (timm `samvit_base_patch16.sa1b`), donuk, 16
  epoch, batch 4, img 512. Pretrained pos-embed/rel-pos native-1024'ten 512 grid'ine interpole edilerek
  yüklendi (sam_backbone.py) → diğer ViT'lerle aynı 32×32 grid. Trunk **256 kanal** (SAM'ın kendi neck'i).
- Feature-caching, fp32 (`--no-amp`). Ağırlık HF Xet CDN'e Colab'dan erişilemediği için PC'den indirilip
  Drive üzerinden `MTL_WEIGHTS_DIR` ile yerel dosyadan yüklendi (models/timm_weights.py).

### Sonuç — altı omurga (hepsi donuk, 16 epoch, batch 4)

| metrik | DINOv2 | ResNet | CLIP | DINOv1 | MAE | **SAM** | SAM sırası |
|---|---|---|---|---|---|---|---|
| detection_mAP | **0.2300** | 0.1965 | 0.1420 | 0.1541 | 0.1336 | 0.1497 | 4/6 |
| seg_mIoU | **0.6011** | 0.3215 | 0.4398 | 0.3928 | 0.2545 | **0.1933** | **6/6** |
| cls_mAP | **0.7800** | 0.7084 | 0.6899 | 0.5565 | 0.4215 | **0.3410** | **6/6** |
| cls_F1 | **0.7239** | 0.6799 | 0.6501 | 0.5515 | 0.4181 | **0.3545** | **6/6** |

### 🎯 Bulgu — "seg-native" model seg'de EN DÜŞÜK
SAM prompt'a karşılık **sınıf-agnostik maske** üretir; **kategori bilgisi öğrenmez**. Bizim seg'imiz
**semantik** (piksel→sınıf). SAM feature'ları nesnelik/sınır taşır ama **kategori semantiği taşımaz**
→ semantik-seg + classification'da (ikisi de kategori ister) en düşük. "seg-native ⇒ seg'de iyi" sezgisi
**çürüdü**: segmentasyonun *türü* belirleyici. (Detection'da SAM 0.150 ile MAE/CLIP üstünde — nesnelik
kutuya biraz yarıyor, ama zayıf sınıflandırma mAP'i sınırlıyor.)

### ⚠️ İki confound (ablasyon gerekli)
1. **256-kanal darboğazı** (SAM neck 768→256) → SFP'ye dar bilgi. Ablasyon: pre-neck 768 tap.
2. **pos-embed/rel-pos interp** (1024→512). Kontrol: native 1024 koşu (ağır).
Mekanizma (kategori-semantiği yokluğu) seg+cls'nin BİRLİKTE düşmesini açıklıyor; 256-darboğaz tek başına
cls'yi bu kadar düşürmezdi → bulgu ayakta ama güçlü iddia için ablasyon şart.

### 🏁 Faz 1 kapandı — altı omurga / beş paradigma
DINOv2 dört metrikte lider; "uzman" pretraining'ler (CLIP=dil, SAM=seg-native, MAE=MIM) donuk rejimde
genel SSL/supervised'ın gerisinde ama her biri kendi ekseninde imzalı. **Tez doğrulandı:** pretraining
sinyali downstream davranışı öngörüyor. Detay ve tam tablo: RESULTS.md "FAZ 1 ÖZET".

### Sonraki adım
- **Faz 2 (adaptasyon):** MAE + DINOv2 çifti, donuk → LoRA → full. Tahmin: MAE en çok kazanır, sıralama değişir.
- **Faz 3 ablasyonları:** SAM pre-neck 768 tap; DINOv2 register/patch; head ekseni (FCOS/Mask2Former).

---

## Kararlar — 2026-07-03 — İkinci omurga olarak DINO ekleniyor

ResNet50+FPN ile yapılan Deneme 1–3'ten sonra, **aynı pipeline'ı omurgada DINO
(self-supervised ViT) kullanarak** ikinci kez koşup iki omurgayı karşılaştırma kararı
alındı. ResNet yolu aynen korunuyor; DINO `backbone_name` ile seçilen alternatif bir
omurga (`configs/train_colab_dino.yaml`). Planlama sırasında verilen kararlar ve gerekçeleri:

- **Backbone / neck / head üç ayrı katman.** FPN bir *neck*'tir, ResNet'e özgü değildir;
  torchvision `resnet_fpn_backbone` sadece gövde+neck'i tek pakette birleştirdiği için
  "backbone"un içinde görünür. Head'ler neck'in **çıktı sözleşmesini** (5 seviye × 256 kanal,
  strides 4/8/16/32/64) tüketir, hangi neck olduğunu umursamaz. Bu yüzden DINO'yu eklemek
  head/loss/eval koduna dokunmadı.
- **Varyant → DINOv1 ViT-B/16.** Patch 16 → stride 16; feature-pyramid stride matematiği
  (4/8/16/32/64) temiz. (DINOv2 patch-14/stride-14 olurdu, ekstra interpolasyon isterdi.)
- **DINO neck'i → gerçek FPN değil, Simple Feature Pyramid (ViTDet, Li et al. 2022).**
  Plain ViT tek çözünürlük (stride 16) üretir; FPN'in birleştireceği doğal hiyerarşi yoktur.
  FPN için multi-block tap + resample ile sahte hiyerarşi uydurmak (ViT-Adapter tarzı) ciddi
  ek karmaşıklık, düz ViT'te belirgin kazanç yok. Simple Feature Pyramid tek stride-16
  haritadan her seviyeyi up/down-sample ile türetir: daha az parametre, T4-dostu, ve neck'i
  minimal tutmak karşılaştırmayı "DINO vs ResNet feature'ları"na odaklar.
- **Eğitim protokolü → config'ten ayarlanabilir; ilk koşu donuk.** `trainable_backbone_layers`
  DINO için "son N eğitilebilir transformer bloğu" anlamına gelir. Önce `0` (tamamen donuk):
  kanonik SSL kullanımı, en az VRAM, ve ResNet Deneme 2'nin donuk-backbone'uyla adil kıyas.
- **Verimlilik tradeoff'u (beklenti).** ResNet hesap/hız/VRAM açısından daha ucuz (~25M,
  konvolüsyonel). DINO ViT-B (~86M, attention O(N²), 512px'te 1024 token) daha ağır — 12 GB
  VRAM'de `batch_size=4` ile başlanacak (config yorumu). Buna karşılık donuk SSL feature'lar
  az-etiketli rejimde (~22.5k subset) güçlü; **beklenti:** DINO en çok classification/segmentation'da
  kazanır, detection'da ResNet'in gerçek FPN'i avantajlı kalabilir. Kesin karar sonuçlarla verilecek.
- **timm tercihi (torch.hub değil).** timm DINO ağırlıklarını taşır, pos-embed
  interpolasyonunu yapar ve `pretrained=False`'ta **offline** kurulur (tests + `eval.py`).
  torch.hub `pretrained=False`'ta bile repo kodunu indirip offline eval/testi bozardı.

### Çıktı formatı değişikliği
Eval çıktısı dağınık `print(metrics)` yerine: (1) hizalı tablo, (2) `runs/results.csv`'ye
tek satır (omurga/checkpoint/adım + 4 metrik). `scripts/compare_results.py` bu CSV'den
ResNet-vs-DINO markdown tablosu üretir (aşağıya kopyalanır).

---

## Karşılaştırma — ResNet vs DINO

**Tüm karşılaştırma sonuçları tek dosyada:** [RESULTS.md](RESULTS.md). Buradaki dağınık
tabloları oraya taşıdık (tek kaynak, birbirinden sapmasın). Ham makine logu `runs/results.csv`.

### ✅ Çözüldü — donuk (layers=0) ResNet koşusu yapıldı (Deneme 4)
Önceden eksikti: DINO `layers=0` (donuk), ResNet'in tüm koşuları `layers=3` (kısmen
eğitilebilir) olduğu için kıyasta **protokol farkı** vardı. (Deneme 2 "layers=0" diye
etiketliydi ama config'i doğrulanmamıştı — güvenilmez.)

**2026-07-04'te donuk ResNet 1 epoch koşuldu** (Deneme 4): det 0.0467 / seg 0.0780 /
cls_mAP 0.4128 / cls_F1 0.4562. Artık "donuk ResNet vs donuk DINO" aynı adımda (2813)
birebir kıyaslanabiliyor → dört metrikte de ResNet önde. Detay ve adil-kıyas tablosu
[RESULTS.md](RESULTS.md)'de (satır 6 + "Adil kıyas" bölümü).
