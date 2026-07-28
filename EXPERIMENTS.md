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

## Deneme 12 — 2026-07-15 — 🏆 DeiT (supervised ViT) → ADİL ÇEKİRDEK TAMAMLANDI

### Kurulum
- `configs/train_colab_deit.yaml`: `deit_vitb16` (timm `deit_base_patch16_224.fb_in1k`, SAF supervised
  in1k — distilled DEĞİL), donuk, 16 epoch, batch 4, img 512. Diğer ViT-B'lerle her eksende aynı.
- Amaç: "supervised" paradigmasını **adil ViT temsilcisiyle** kapatmak → ResNet'in conv/FPN confound'u kalksın.

### Sonuç (donuk, 16 epoch, batch 4)
det 0.1369 / seg 0.4271 / cls_mAP 0.6871 / cls_F1 0.6513. (AP@0.50=0.278; small 0.028 / med 0.133 / large 0.255.)

### 🎯 Bulgular
1. **DINOv1 (SSL) vs DeiT (supervised) — en temiz kıyas:** DeiT, DINOv1'i **cls'de açık ara** (0.687 vs
   0.557) **ve seg'de** (0.427 vs 0.393) geçer; DINOv1 sadece **det'te hafif önde** (0.154 vs 0.137). →
   Önceki "ResNet supervised cls/det'te güçlü" **conv/FPN artefaktı değilmiş**; supervised ViT de tanıma-güçlü.
2. **DeiT ≈ CLIP** (cls ~0.69, seg ~0.43, det ~0.14): supervised ve dil, ikisi de "görüntüde ne var" için
   optimize → aynı tanıma-güçlü/lokalizasyon-orta imza.
3. **Ama DINOv2 hepsini geçiyor** → "supervised > SSL" DINOv1'e özgü; pretraining kalitesi (DINOv2) artınca
   SSL öne geçiyor. Ana mesaj: paradigma değil, **kalite+tür** belirleyici.
4. **Detection deseni:** tüm ViT-B'ler 0.13–0.15 kümesinde; yüksekler DINOv2 (ince grid) + ResNet (FPN) →
   detection farkı büyük ölçüde grid/neck kaynaklı, Faz 3 işareti.

### Adil çekirdek tamam
Beş paradigma, hepsi ViT-B/16 @512: DeiT (supervised) · DINOv1 (SSL-distill) · MAE (MIM) · CLIP (dil) ·
SAM (seg-native). + bağlam (ResNet, DINOv2) + dipnot (I-JEPA). Tam tablo/bulgu: RESULTS.md.

---

## Deneme 11 — 2026-07-15 — 📎 I-JEPA ViT-H (DİPNOT): "boyut ≠ kalite"

### Kurulum
- `configs/train_colab_ijepa.yaml`: `ijepa_vith16` (HF transformers), donuk, 16 epoch, batch 4, img 512.
- ⚠️ **ViT-H (632M)** — Meta ViT-B yayınlamadı → **boyut confound'u düzeltilemez** → ana kıyasa değil,
  **DİPNOT**. patch16@512 seçilerek grid+batch confound'ları kaldırıldı; boyut kaldı.
- Bug düzeltmesi: eval `pretrained=False`'ta default (ViT-B) config kuruyordu → `IJepaConfig.from_pretrained`
  ile ViT-H mimarisi kurulur hale getirildi (ijepa_backbone.py).

### Sonuç (donuk, 16 epoch, batch 4)
det 0.1941 / seg 0.4264 / cls_mAP 0.6149 / cls_F1 0.6047. (AP@0.50=0.357; small 0.057 / med 0.203 / large 0.335.)

### 🎯 İki bulgu (boyut confound'una rağmen anlamlı)
1. **7× büyük olmasına RAĞMEN I-JEPA DINOv2'yi (ViT-B) dört metrikte de geçemiyor.** → "büyük model ≠
   iyi feature"; **pretraining kalitesi ham boyutu yeniyor.** Confound'un *aleyhine* kanıt: ölçek her şey
   olsaydı ViT-H lider olurdu, olmadı.
2. **I-JEPA >> MAE her metrikte** (seg 0.43 vs 0.26; cls 0.61 vs 0.42). Maskeli-tahmin ailesinde **latent
   tahmin (I-JEPA) piksel-kurmayı (MAE) geçiyor** — I-JEPA'nın ana iddiası. ⚠️ boyut confound'lu, yön
   tutarlı ama kesin değil.

### ⚠️ ÇİFT confound (RESULTS.md dipnotunda detaylı) — I-JEPA aleyhine
Boyutun (ViT-H) yanında **ikinci** bir confound: neck sıkıştırması. embed_dim 1280 → SFP 256 = **5×**;
ViT-B'ler 768 → 256 = **3×** → I-JEPA daha sert sıkışıyor, zengin feature'ı dar neck'te boğulabilir. Bu
"boyut≠kalite"yi hafifçe zayıflatır (geniş neck'le kazanabilirdi). Karşı: neck 1×1 EĞİTİLİYOR → kayıp
sınırlı. Tasarım gerilimi: embed_dim farklıysa "neck genişliği sabit" ve "sıkıştırma oranı sabit" birlikte
tutulamaz → adil çekirdek (hepsi 3×) temiz, I-JEPA/SAM dipnot. Temiz test: geniş neck (1280→512) / ViT-B I-JEPA.

### Faz 1 + dipnot kapandı
Yedi omurga (ResNet, DINOv1, DINOv2, CLIP, MAE, SAM adil/bağlam + I-JEPA dipnot). DINOv2 her yerde lider;
en büyük+predictive model bile onu geçemedi. Detay: RESULTS.md.

---

## Deneme 13 — 2026-07-17 — ⚠️ BEiT donuk @512 BAŞARISIZ (açık soruşturma, ana tabloya alınmadı)

### Kurulum
- `configs/train_colab_beit.yaml`: `beit_vitb16` = HF `microsoft/beit-base-patch16-224-pt22k` (**saf SSL
  pretrain**, fine-tune DEĞİL), donuk, 16 epoch, batch 4, img 512. Maskeleme ailesinin "masked-token" ayağı.
- Cache'li akış (precompute → train_cached --no-amp → eval).

### Sonuç (donuk, 16 epoch, batch 4) — FELAKET
det **0.033** / seg **0.061** / cls_mAP **0.232** / cls_F1 **0.249**. Diğer tüm omurgaların 5-10× altında;
en kötü diğer (SAM seg 0.193) bile çok üstünde. BEiT 16 epoch sonunda ≈ DINO'nun **yarım epoch**'u (Deneme 6,
step 2813: seg 0.056 / cls 0.233) → "zayıf ön-eğitim" değil, **eğitilmemiş/bozuk seviye.**

### 🔍 İki şey soruşturuldu

**1. Düşen final LayerNorm → devasa aktivasyonlar (GERÇEK ama sebep DEĞİL).**
BEiT `use_mean_pooling=True` config'inde final LayerNorm'u pooler'a taşır; biz `add_pooling_layer=False` ile
pooler'ı kapattığımızdan `last_hidden_state` normalize edilmemiş döndü → cache feature'ları std ~19, aralık
±500 (yükleme raporundaki `layernorm UNEXPECTED` bunun habercisiydi). Çözüm: `_trunk_raw`'da parametresiz
`F.layer_norm` (beit_backbone.py). Cache std 1.0'a indi. **AMA yeniden eğitim BİREBİR aynı sonucu verdi
(0.033/0.061/0.232)** → ölçek sorun değildi. Düzeltme yine de doğru (bırakıldı), ama başarısızlığı açıklamıyor.

**2. Rel-pos bias ekstrapolasyonu @512 (ÖNDE GELEN HİPOTEZ, henüz kanıtlanmadı).**
Feature'lar çökmemiş, iyi ölçekli, görselleri ayırıyor — ama transfer etmiyor → konumsal yapı bozuk olabilir.
BEiT **absolute pos-embed KULLANMAZ**, bunun yerine **relative-position-bias** kullanır: "iki patch N adım
uzaksa dikkate şu biası ekle" kural tablosu. 224'te (14×14) tablo yalnız **±13** adıma kadar öğrenildi; 512'de
(32×32) patch'ler **±31** adım uzak olabilir → bu mesafeler eğitimde HİÇ görülmedi → interpolasyon aslında
**ekstrapolasyon** (uydurma). Üstelik rel-pos **her katmanda tekrar** eklenir → hata katmanlar boyunca birikir;
absolute pos (girişte bir kez, DINO/MAE/CLIP) gibi toparlanamaz. Bu yüzden BEiT çözünürlüğe diğerlerinden
çok daha hassas. (Kıyas: SAM 1024→512 = küçültme/interpolasyon, güvenli yön → çalıştı; BEiT 224→512 =
büyütme/ekstrapolasyon, tehlikeli yön → çalışmadı.)

### 📄 Makale çelişkisi değil — fine-tuning farkı
BEiT makalesi (Ek B, Tablo 6) 512²'de sonuç raporlar, AMA orada model **512'de fine-tune EDİLİR** ("intermediate
fine-tuning ... evaluate at 384²/512²"). Fine-tune sırasında rel-pos bias **gradyan alıp 512 grid'ine uyum
sağlar** — interpolasyon sadece başlangıç noktasıdır. Bizde backbone **donuk** → o bozuk rel-pos hiç düzeltilemez.
Yani makale "512'de **eğitirsen** çalışır", biz "512'de **dondurursan** çalışmaz" diyoruz — çelişki yok.

### ⚠️ Dürüstlük: sebep KESİN değil
Rel-pos hipotezi güçlü ve makaleyle tutarlı ama **kanıtlanmadı**. Muhtemelen iki etki üst üste biniyor:
(a) BEiT donuk-feature'ı zaten zayıf (MAE gibi, bilinen linear-probe zayıflığı) + (b) 512 rel-pos ekstrapolasyonu
onu felakete çeviriyor. Kesin ayrım tek deneyde.

### Sonraki adım — belirleyici test
**BEiT'i donuk olarak native 224'te koş** (14×14 grid, interpolasyon YOK):
- 224'te düzgün gelirse (~MAE civarı) → suçlu **kesinlikle 512/rel-pos**; grid-confound'lu ama geçerli bir
  BEiT sayısı elde edilir (dipnotla ana tabloya girer).
- 224'te de bozuksa → sorun çözünürlük değil; BEiT donuk-feature'ı bu görevler için genel zayıf, ya da çıkarım
  kodunda başka sorun → oraya bakılır.

Gerekli: küçük kod tweak'i (backbone h,w'yi girdi şeklinden hesaplasın, sabit 512 yerine) + `train_colab_beit224.yaml`.

### GÜNCELLEME 1 (2026-07-17) — belirleyici test: ÇÖZÜNÜRLÜK HİPOTEZİ REDDEDİLDİ
BEiT native 224 (donuk, 16 epoch, 14×14 grid, interpolasyon YOK): **det 0.026 / seg 0.068 / cls_mAP 0.271
/ cls_F1 0.232** — 512'yle (seg 0.061) **neredeyse aynı** → **rel-pos/çözünürlük hipotezi YANLIŞ.** 224'te
feature'lar native (temiz) ama BEiT yine çöküyor. Suçlu çözünürlük değilmiş.

### 🎯 GÜNCELLEME 2 (2026-07-17) — asıl suçlu bulundu: NORMALİZASYON UYUŞMAZLIĞI
Terminoloji düzeltmesi (kullanıcı uyarısı): biz **linear probe YAPMIYORUZ** — donuk backbone + **eğitilebilir
SFP neck + eğitilebilir head'ler** (FCN/RetinaNet/cls, non-linear decoder). Bu, linear probe'dan güçlü. Ve tam
da bu, sorunu açığa çıkardı: eğitilebilir head'lerimiz **zayıf-donuk MAE'den seg 0.255** çekebiliyor ama
**BEiT'ten sadece 0.068** → fark "biraz daha zayıf"la açıklanamayacak kadar büyük → **BEiT'e özgü GİRDİ sorunu.**

**Kök neden:** HF `BeitImageProcessor` girdiyi **mean/std = [0.5, 0.5, 0.5]** ([-1,1]) ile bekler; biz dataset'te
**tüm** backbone'lara **ImageNet norm** veriyoruz. MAE (timm) ImageNet ister → doğru; **BEiT (HF) 0.5 ister →
yanlış besleniyor** → feature'lar sistematik kayar (geçerli görünür ama bozuk). Bu hipotez HER ŞEYİ açıklar:
çözünürlükten bağımsız (224=512 ✓), BEiT'e özgü (✓), MAE'yle orantısız fark (✓), feature dejenere değil ama
kötü (✓). CLIP'te bu renorm'u yapmıştık; **BEiT'te atlamışız** (docstring yanlışlıkla "ImageNet, renorm gerekmez"
diyordu).

**Fix:** `beit_backbone.py`'ye ImageNet→BEiT(0.5) renorm buffer'ları eklendi (CLIP deseni). **Doğrulama:**
`BeitImageProcessor.from_pretrained(...).image_mean/std == [0.5]*3`. Sonra cache'i yeniden hesapla + eğit + eval.

**Durum:** BEiT sonucu HÂLÂ AÇIK — 0.068 sayısı **yanlış-normalize** koşudan; RESULTS.md ana tablosuna
**henüz alınmadı.** Renorm'lu re-run gerçek BEiT sayısını verecek. (Önceki "BEiT gerçekten en zayıf" çıkarımı
GERİ ÇEKİLDİ — muhtemelen normalizasyon hatasıydı; üçüncü hipotez, ilk ikisi de yanlıştı.)

---

## Deneme 14 — 2026-07-17 — ⚡ Verim ölçümü (FPS / gecikme / bellek, A100)

`scripts/benchmark_latency.py` ile 8 backbone, **batch=1** (gerçek-zaman), **A100-SXM4-40GB**. Ölçülen:
backbone + SFP neck ileri-geçişi. (params SFP dahil → ViT-B ~91M.) Tam tablo: RESULTS.md "Verimlilik".

**FPS:** ResNet **91** · ViT-B dörtlü (DeiT/DINOv1/CLIP/MAE) **~62** · DINOv2 **45** · SAM **36** · I-JEPA(ViT-H) **9.3**.
**Bellek:** ResNet 237 MB → I-JEPA **2729 MB (11×)**. **Param:** ViT-B ~91M → I-JEPA **642M (7×)**.

**Üç bulgu:**
1. **ViT-B @512 dörtlüsü maliyette BİREBİR AYNI** (~62 FPS, 447 MB) → **pretraining objektifi "bedava eksen"**:
   doğruluk farkları sıfır ek maliyetle. Adil çekirdeğin maliyet-eşitliğini de kanıtlar (aynı mimari = aynı maliyet).
2. **ResNet = verimlilik kralı** (91 FPS, 27M, 237MB) → kısıtlı platform baseline'ı olarak neden güçlü.
3. **I-JEPA ViT-H = ölçek vergisi**: en yavaş (9.3 FPS), en çok bellek (2.7 GB), 642M — **VE doğruluk lideri
   değil** (DINOv2 ViT-B geçiyor). "boyut≠kalite" (Deneme 11) + "boyut pahalı" birleşiyor: doğruluk↔FPS
   grafiğinde I-JEPA sağ-altta (yavaş+vasat), DINOv2 sağ-üstte (hızlıca iyi) → motivasyona (kısıtlı platform)
   doğrudan bağlanan en güçlü tek grafik.

⚠️ Not: BEiT verimini tabloya koymadım — eğitim sonucu hâlâ açık (Deneme 13). Mimari FPS'i geçerli ama
sonuç netleşince eklenecek.

---

## Deneme 16 — 2026-07-27 — 🔬 FAZ 3: adaptif loss (Kendall) → "kazanç değil YENİDEN DAĞITIM"

### Kurulum
- `configs/train_colab_dinov2_adaptive.yaml`: donuk DINOv2 (Deneme 7) ile **TEK FARK** loss —
  sabit `1/1/1/0.5` yerine **öğrenilen belirsizlik ağırlıkları** (Kendall 2018, `UncertaintyWeighter`:
  görev başına öğrenilebilir `log σ²`, `L = Σ exp(-sᵢ)Lᵢ + sᵢ`). Feature-caching, aynı 22.5k/seed/lr.
- Not: eğitimde **total loss negatife iniyor** — beklenen (log-var regularizer terimi `+sᵢ`, düşük-loss
  görevlerde negatif). Kalite ölçüsü total değil, per-task loss + eval.

### Sonuç — DINOv2 sabit-loss vs adaptif-loss (tek değişken: ağırlıklandırma)

| metrik | sabit (Deneme 7) | adaptif | delta |
|---|---|---|---|
| detection_mAP | 0.2300 | 0.2155 | **−6%** |
| seg_mIoU | 0.6011 | 0.5971 | −0.7% |
| cls_mAP | 0.7800 | 0.8053 | **+3.3%** |
| cls_F1 | 0.7239 | 0.7534 | **+4.1%** |

**Öğrenilen ağırlıklar exp(-s):** cls_loss **29.0** ≫ seg 5.9 > det_cls 3.6 > bbox 2.5.

### 🎯 Bulgu — uniform iyileşme YOK, kapasite yeniden dağıtıldı
`exp(-sᵢ) ≈ 1/Lᵢ` (optimumda `sᵢ = log Lᵢ`) → **düşük-magnitüdlü loss yüksek ağırlık alır.** cls_loss zaten
minik (~0.05) → model onu **29×** yukarı çekti (sabit 0.5'e karşı). Etki: **cls yukarı (+%3-4), detection
aşağı (−%6), seg sabit** → adaptif ağırlıklandırma **kazanç değil yeniden-dağıtım.** Üstelik "yanlış" yönde:
**kolay görevi (cls) besledi, en zor görevi (detection) aç bıraktı** — çünkü Kendall şeması görevleri
*magnitüde göre* dengeler, *zorluğa göre* değil.

**Yorum:** sabit `1/1/1/0.5` bu kurgu için zaten makul dengedeymiş; adaptif loss'un asıl etkisi küçük-magnitüdlü
cls loss'unu normalize edip yukarı çekmek oldu. **Meşru bir negatif/nötr bulgu:** her ablasyon iyileştirmez;
bu, mevcut sabit ağırlıkların savunulabilir olduğunu da gösterir. (İstenirse: cls'i elle down-weight'lemek
yerine detection'ı upweight'leyen manuel ablasyon; ya da GradNorm/PCGrad — Faz 3 devamı.)

**Faz 3 açıldı** (neck-PANet ablasyonu kod olarak hazır, koşulmadı; loss ablasyonu bu Deneme).

---

## Deneme 17 — 2026-07-28 — 🔬 FAZ 3: task-native neck → "donuk tavanda kazanç yok; çözünürlük > bağlam"

### Kurulum
- `configs/train_colab_dinov2_taskneck_native.yaml`: donuk DINOv2 (Deneme 7) ile **tek eksen = neck yapısı**.
  Paylaşılan tek SFP yerine her göreve **native neck**, hepsi doğrudan ham donuk trunk'tan:
  **det=SFP** (piramit, RetinaNet 5-seviye sözleşmesi zorunlu), **seg=ASPP** (dense context, SFP dalı YOK),
  **cls=GAP+Linear** (ham 768-boyutlu trunk, SFP dalı YOK).
- Kod: `neck_mode` enum'u (`shared` | `per_task_identical` | `task_native`; config.py + multitask_model.py,
  commit c269321). Baseline `shared` bozulmadı; `_run_heads` değişmedi (seg/cls head'leri embed_dim girdiyle kurulur).
- Trunk donuk ve AYNI → **DINOv2 baseline cache'i paylaşıldı, precompute YOK.** Sabit loss `1/1/1/0.5`,
  aynı 22.5k/seed/lr. Feature-caching, `--no-amp`. Checkpoint: `colab_dinov2_taskneck_native_cached_epoch15.pt`.

### Sonuç — shared (Deneme 7) vs task_native (tek değişken: neck)

| metrik | shared (Deneme 7) | task_native | delta |
|---|---|---|---|
| detection_mAP | 0.2300 | 0.2277 | −1.0% |
| seg_mIoU | 0.6011 | 0.5951 | −1.0% |
| cls_mAP | 0.7800 | 0.7947 | **+1.9%** |
| cls_F1 | 0.7239 | 0.7318 | +1.1% |

(Verim: 92.1M / 45.4 FPS / 548 MB — benchmark donuk backbone+SFP forward'ını ölçer, baseline DINOv2 ile aynı;
native neck'lerin ekstra parametresi head tarafında, bu ölçümde görünmez.)

### 🎯 Bulgu — donuk-tavanda anlamlı kazanç YOK (ROADMAP tahmini doğrulandı)
Farklar ±1% bandında; net etki küçük bir **yeniden-dağıtım** (cls hafif ↑, det/seg hafif ↓). ROADMAP'in
"donuk DINOv2 zaten tavana yakın (seg 0.60) → ölçülebilir kazanç çıkmayabilir; **negatif de geçerli bulgu**"
beklentisi tuttu. Üç mekanizma, hepsi tutarlı:
1. **cls +%1.9 (tek pozitif sinyal, temiz mekanizma):** native cls neck GAP'i **ham 768-d trunk'a** uygular;
   baseline SFP trunk'ı **256-kanala + stride-32'ye sıkıştırıp** öyle pool'lar → sıkıştırılmamış zengin
   feature görüntü-sınıflandırmaya küçük ama tutarlı fayda. Beklenen yönde.
2. **seg −%1.0 (ASPP yardım ETMEDİ):** kaba trunk grid'inde (37×37) ASPP + ×14 upsample, baseline'ın
   SFP-level0'ından (stride-4, çok daha ince) + FCN'inden **sınır olarak daha kaba**. Bağlam kazancı
   çözünürlük kaybıyla siliniyor → **alt-bulgu: donuk DINOv2 seg için çözünürlük > bağlam.** (DeepLabv3
   vs v3+ noktası: saf ASPP + büyük upsample sınır-kaba.)
3. **det ~sabit:** iki modda da det=SFP; tek fark ayrı vs paylaşılan ağırlık → ihmal edilebilir.

**Kıyas (Deneme 16 ile):** adaptif loss da cls'i yukarı, det'i aşağı çekmişti (yeniden-dağıtım). Burada da
benzer desen ama **çok daha küçük magnitüd** ve **farklı mekanizma** (mimari, ağırlıklandırma değil).

### ⚠️ Kapsam (confound) — saf interference DEĞİL
Çok-değişkenli: seg dalı hem **ayrıştı** hem **decoder değişti** (SFP+FCN → ASPP). Bu yüzden "interference
azaldı" **DENEMEZ**; yalnız *"native mimari neck bu donuk-tavan rejiminde kazandırmıyor"* denebilir. Saf
interference için `per_task_identical` (B: aynı SFP dağarcığı, tek değişken = paylaşım) gerekli.

### Sonraki adım
- **`per_task_identical` (B) koşusu** → **A→B→D merdiveni**: "ayırmak" (paylaşılan→özdeş SFP) ile "native
  mimari" (özdeş→ASPP/GAP) katkıları ayrışır.
- Opsiyonel: Faz 2'de **çözülen** backbone'da (LoRA) native neck etkisi farklı olabilir — orada tavan yok.

---

## Deneme 15 — 2026-07-25 — 🎯🎯🎯 FAZ 2 AÇILDI: MAE + LoRA → "donukken kötü, çözüldüğünde harika" DOĞRULANDI

### Kurulum
- `configs/train_colab_mae_lora.yaml`: donuk MAE (Deneme 9) ile **TEK FARK** LoRA — taban ViT donuk,
  yalnız LoRA adaptörleri (rank 8, alpha 16, qkv+proj, 12 blok) + neck + head eğitilir.
- **AYNI 22.500 görüntü, aynı val, aynı lr/wd/seed/loss/aug** → donuk baseline ile birebir kıyaslanabilir,
  **delta geçerli** (veri confound'u YOK). Cache YOK (LoRA trunk'ı değiştirir) → normal `train.py`.
- Arkadaşın workstation'ında koşuldu (yerel kurulum: setup_local.sh + setup_data.sh).

### Sonuç — donuk MAE vs MAE+LoRA (aynı veri, tek fark adaptasyon)

| metrik | MAE donuk (Deneme 9) | **MAE + LoRA** | delta |
|---|---|---|---|
| detection_mAP | 0.1336 | **0.2394** | **+79%** |
| seg_mIoU | 0.2545 | **0.4640** | **+82%** |
| cls_mAP | 0.4215 | **0.6388** | +52% |
| cls_F1 | 0.4181 | **0.6180** | +48% |

(COCO: AP@0.50=0.399, AP@0.75=0.247; small 0.099 / med 0.247 / large 0.384. Verim arkadaşın GPU'sunda:
91.8M param [+0.4M LoRA adaptörü], 50.3 FPS — A100 tablosuyla kıyaslanamaz, farklı donanım.)

### 🎯 Bulgu — Deneme 9 tahmini DOĞRULANDI
Donuk sweep'in **DÖRT METRİKTE SONuncusu** olan MAE, LoRA ile çözülünce **patladı** — det+seg %80 sıçradı.
Dahası: **MAE-LoRA detection'da DINOv2-donuk'u (0.230) geçiyor** (0.239) → *en kötü donuk model, çözülünce
en iyilerden birine.* MAE'nin bilinen imzası (*"linear-probe %68 zayıf, fine-tune %83.6 SOTA"*) bizim
multi-task/donuk→LoRA ekseninde **birebir tekrarlandı.** Piksel-yeniden-kurma semantiği donukken saklıyor;
küçük bir adaptasyon (LoRA) onu açığa çıkarıyor.

### ✅ TAMAMLANDI (2026-07-27) — DINOv2-LoRA geldi, tahmin DOĞRULANDI
DINOv2+LoRA (aynı 22.5k, aynı LoRA ayarı): **det 0.2662 / seg 0.6036 / cls_mAP 0.8156 / cls_F1 0.7787**
(AP@0.50=0.445). İki-delta kontrastı:

| metrik | MAE delta (zayıf-donuk) | DINOv2 delta (güçlü-donuk) |
|---|---|---|
| detection | **+79%** | +16% |
| **seg_mIoU** | **+82%** | **+0.4%** (0.601→0.604, tavan) |
| cls_mAP | +52% | +4.6% |
| cls_F1 | +48% | +7.6% |

**🎯 Bulgu — adaptasyon kazancı donuk-kaliteyle TERS ORANTILI (tahmin doğrulandı):** zayıf-donuk MAE
LoRA'dan **devasa** (+%80), güçlü-donuk DINOv2 **çok az** (seg neredeyse sıfır — zaten tavanda) kazandı.
En temiz kanıt seg: DINOv2 0.601→0.604 (headroom yok) vs MAE 0.255→0.464 (+%82). → *pretraining sinyali
sadece "hangi görevde iyi"yi değil, **"hangi ADAPTASYON rejiminde iyi"yi de öngörüyor.***

**⚠️ Nüans (tam sıralama flip'i DEĞİL):** DINOv2 **LoRA'da da dört metrikte lider** (det 0.266>0.239,
seg 0.604>0.464) — MAE↔DINOv2 arası sıralama değişmedi; DINOv2 hem donuk hem LoRA'da en iyi. Değişen:
(a) **uçurum çöktü** (MAE yaklaştı), (b) **MAE genel sıralamada zıpladı** — donuk sonuncuyken LoRA'da
tüm donuk ViT-B'leri (DINOv1/CLIP/DeiT) ve ResNet/DINOv2-donuk'u geçti. Yani "en iyi donuk = en iyi
fine-tuned" (DINOv2) ama **kazanç büyüklüğü donuk kaliteyle ters** — "donuk feature kalitesi ≠ backbone
kalitesi" mesajı kanıtlandı.

**Faz 2 kapandı** (MAE + DINOv2). Verim: DINOv2-LoRA 92.6M / 43.2 FPS / 502MB (arkadaşın GPU'su, A100
tablosuyla kıyaslanamaz).

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
