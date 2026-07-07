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
