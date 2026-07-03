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

### ⚠️ Düzeltme — donuk (layers=0) ResNet hiç koşulmadı
Yukarıdaki **Deneme 2**, `trainable_backbone_layers=0` (donuk backbone) diye kayıtlı ama
o koşunun config'i doğrulanmadı (Deneme 3 notundaki flip-flop uyarısına bakınız). Kullanıcı
teyidi: **ResNet hiçbir zaman layers=0 ile çalıştırılmadı** — yani güvenilir bir donuk-ResNet
sonucu yok. Şu ana kadarki ResNet koşularının hepsi **layers=3** (kısmen eğitilebilir):
step 200 (Deneme 1), 2813 (Deneme 3), 4500. DINO ise **layers=0** (donuk).

Sonuç: mevcut kıyasta backbone farkının yanında **protokol farkı** da var (kısmen-eğitilebilir
ResNet vs donuk DINO). Birebir adil kıyas için eksik koşu — donuk ResNet — yapılmalı:
`--overrides model.trainable_backbone_layers=0`. Detay ve boş tablo hücreleri [RESULTS.md](RESULTS.md)'de.
