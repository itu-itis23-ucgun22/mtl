# MTL — Kapsamlı Proje Özeti (her şey)

> Bu dosya projenin **tek noktada tam haritası**: yapılan tüm deneyler, bulgular, uygulanan tüm kod,
> düşünülüp uygulanmayan fikirler, ve gelecek işler. Kaynaklar: [EXPERIMENTS.md](EXPERIMENTS.md) (deney
> günlüğü, Deneme 1–23), [RESULTS.md](RESULTS.md) (sonuç tabloları), [ROADMAP.md](ROADMAP.md) (plan),
> [ARCHITECTURE.md](ARCHITECTURE.md) (mimari) + kod. Not: bu belge tek bir Claude oturumunda derlendi;
> önceki ayrı sohbet oturumlarına erişim yoktu, ama MD dosyaları tüm geçmişi kaydettiği için tam.

---

## 0. Tek cümlede proje + tez + motivasyon

**Bir paylaşılan (donuk foundation) backbone + tek neck + üç göreve özel head** ile aynı anda
**detection (RetinaNet) + semantic segmentation (FCN) + multi-label classification (GAP+FC)**, ~22.5k
COCO subset'te.

- **Tez:** *Pretraining sinyali (supervised / SSL / dil / seg-native / MIM), az-etiketli çok-görevli
  dense tahminde hangi downstream görevde iyi olacağını **öngörür mü?***
- **Motivasyon:** kısıtlı platform (uçak/edge) → **3 ayrı küçük model mi, yoksa 1 paylaşılan backbone +
  3 head mi?** Maliyet-farkındalıklı karşılaştırma (3× backbone vs 1× backbone).
- **Protokol (kanonik):** donuk backbone · 16 epoch · batch 4 · img 512 (patch14'te 518) · seed 42 ·
  sabit loss (det_cls 1 / det_box 1 / seg 1 / cls 0.5) · her deney bundan **tek şey** değiştirir.

---

## 1. Mimari (özet)

```
görüntü → [gövde + neck] → features {"0".."3","pool"} (256 kanal) ─┬→ DetectionHead (RetinaNet) → cls+box loss
                                                                    ├→ SegHead (FCN/ASPP) → seg loss
                                                                    └→ ClsHead (GAP+FC) → cls loss
                                                              → joint_loss → tek backward
```

- **Neck takılabilir:** ResNet→FPN (torchvision), ViT'ler→Simple Feature Pyramid (ViTDet, tek grid'den
  5 seviye). Head'ler sadece "5-seviye × 256ch" sözleşmesini tüketir → backbone değiştirmek head/loss/eval'e
  dokunmaz.
- **Neden RetinaNet (Mask R-CNN değil):** RetinaNet `backbone→dict→head` sınırını temiz bırakır; 3 head aynı
  dict'i okur. (Bu yüzden seg **semantic**, instance değil — bilinçli v1 kararı; `maskrcnn_v2.py` stub.)
- **Config sistemi:** dataclass (data/model/loss/train) + YAML + CLI override. Aynı config lokal smoke +
  Colab GPU'yu sürer.
- **Feature-caching (donuk sweep varsayılanı):** donuk trunk deterministik → bir kez `precompute_features`
  ile diske yaz → `train_cached` sadece neck+head'i cache'ten eğit (pahalı ViT forward atlanır). Kod:
  `trunk_forward`/`neck_forward` + `MultiTaskModel.forward_from_trunk`. ResNet donuk gövdesi ucuz → cache yok.

---

## 2. Deney kronolojisi (Faz 0 → Faz 3), tüm 23 Deneme

### FAZ 0 — Protokol kuruluşu (Deneme 1–4)
| # | kurulum | det/seg/cls | ders |
|---|---|---|---|
| 1 | ResNet layers=3, 200 adım | 0.002/0.021/0.130 | pipeline uçtan-uca çalışıyor; azlık, bug değil |
| 2 | layers=0 ilk deneme | (güvenilmez) | config doğrulanmadı → tabloya alınmadı |
| 3 | ResNet layers=3, 1 epoch (2813) | 0.052/0.122/0.429 | 14× adımda hepsi belirgin iyileşti → trend sağlıklı |
| 4 | donuk ResNet layers=0, 1 epoch | 0.047/0.078/0.413 | ilk "donuk vs donuk" (yarım-epoch, confound'lu) |

**Ders:** kısa koşulardan erken sonuç çıkarmak yanıltıcı → eşit epoch + eşit batch + donuk şart.

### FAZ 1 — Backbone sweep (donuk, 16 epoch, batch 4) ⭐ ÇEKİRDEK
Adil çekirdek: hepsi ViT-B/16 @512 → 32×32 grid, **tek değişken pretraining**.

| # | backbone | paradigma | det | seg | cls_mAP | cls_F1 | imza |
|---|---|---|---|---|---|---|---|
| 6 | ResNet50 | supervised (conv) | 0.1965 | 0.3215 | 0.7084 | 0.6799 | İLK ADİL KIYAS (satır 8) |
| 5 | DINOv1 | SSL-distill | 0.1541 | 0.3928 | 0.5565 | 0.5515 | seg-eğilimli/zayıf (satır 7) |
| 7 | **DINOv2** | SSL v2 | **0.2300** | **0.6011** | **0.7800** | **0.7239** | 🏆 DÖRT METRİKTE LİDER (satır 9) |
| 8 | CLIP | dil-contrastive | 0.1420 | 0.4398 | 0.6899 | 0.6501 | semantik-güçlü/lokalizasyon-zayıf (satır 10) |
| 9 | MAE | masked-pixel (MIM) | 0.1336 | 0.2545 | 0.4215 | 0.4181 | DÖRT METRİKTE SON — "donukken kötü" (satır 11) |
| 10 | SAM | seg-native | 0.1497 | **0.1933** | 0.3410 | 0.3545 | seg'de EN DÜŞÜK (class-agnostic ≠ semantik) (satır 12) |
| 12 | DeiT | supervised (ViT) | 0.1369 | 0.4271 | 0.6871 | 0.6513 | supervised adil temsilci ≈CLIP (satır 14) |
| 11 | I-JEPA ViT-H | predictive-SSL | 0.1941 | 0.4264 | 0.6149 | 0.6047 | 📎 DİPNOT (boyut confound; DINOv2'yi geçemedi) (satır 13) |
| 13 | BEiT | masked-token | (geçersiz) | | | | ⚠️ normalizasyon hatası — AÇIK, tabloda yok |

**Deneme 14 — Verimlilik** (A100, batch=1, backbone+SFP): ResNet **91 FPS**/27M/237MB · ViT-B dörtlüsü
(DeiT/DINOv1/CLIP/MAE) **~62 FPS**/91M/447MB (maliyette birebir aynı → "pretraining bedava eksen") · DINOv2
45 FPS/92M/474MB · SAM 36 FPS · I-JEPA ViT-H **9.3 FPS**/642M/2.7GB ("ölçek vergisi").

**Faz 1 bulguları:**
1. **DINOv2 dört metrikte lider.** Genel-amaçlı güçlü SSL her yerde kazanıyor.
2. **"Uzman" pretraining'ler donukta zayıf** ama her biri **görev-imzalı:** CLIP (dil) semantik-güçlü/
   lokalizasyon-zayıf; MAE (MIM) donukken kötü/düşük-seviye-güçlü (small AP en yüksek 0.048); SAM (seg-native)
   kategori-semantiği yok → seg+cls en düşük.
3. **En temiz kıyaslar:** CLIP vs DINOv1 (dil vs SSL, aynı grid) · DeiT vs DINOv1 (supervised vs SSL, aynı ViT).
4. **I-JEPA ViT-H (7×) bile DINOv2 ViT-B'yi geçemedi** → "boyut ≠ kalite". I-JEPA >> MAE (latent > piksel tahmini).
5. **Tez doğrulandı:** pretraining sinyali downstream davranışı öngörüyor; "genel" sinyal (DINOv2) geniş,
   "uzman" sinyaller dar transfer.

### FAZ 2 — Adaptasyon: LoRA (Deneme 15)
Donuk → LoRA (rank8, qkv+proj), iki uç: MAE (zayıf) + DINOv2 (güçlü). Aynı 22.5k, tek fark adaptasyon.

| backbone | det | seg | cls_mAP | cls_F1 | donuğa göre |
|---|---|---|---|---|---|
| MAE+LoRA (satır 16) | 0.2394 | 0.4640 | 0.6388 | 0.6180 | **det +%79 / seg +%82** |
| DINOv2+LoRA (satır 17) | 0.2662 | 0.6036 | 0.8156 | 0.7787 | det +%16 / **seg +%0.4 (tavan)** / cls +%5-8 |

**🎯 Yasa:** **adaptasyon kazancı ∝ 1/donuk-kalite.** Zayıf-donuk MAE devasa (+%80), güçlü-donuk DINOv2 çok
az (tavan). Deneme 9'un tahmini doğrulandı. *Pretraining sinyali "hangi görevde iyi"yi değil, "hangi
ADAPTASYON rejiminde iyi"yi de öngörüyor.* ⚠️ Tam sıralama flip'i DEĞİL — DINOv2 LoRA'da da lider; ama uçurum
çöktü, MAE donuk-sonuncudan LoRA'da tüm ViT-B'leri geçmeye zıpladı.

### FAZ 3 — Ablasyonlar (Deneme 16–23)
| # | değişiklik | sonuç | bulgu |
|---|---|---|---|
| 16 | Adaptif loss (Kendall) | cls +3-4% / det −6% / seg sabit | **kazanç değil yeniden-dağıtım**; küçük-magnitüd cls'i besledi, zor det'i aç bıraktı |
| 17 | Task-native neck (det=SFP/seg=ASPP/cls=GAP) | ~düz (cls +1.9%) | neck oynatmak DINOv2 seg/cls'ini iyileştirmedi; alt-bulgu **çözünürlük > bağlam** |
| 18 | **ResNet + ASPP** (seg decoder) | **seg 0.32→0.46 (+%43)** 🎯 | seg zayıflığı **DECODER'mış, feature değil**; DINOv2-ASPP ile tam zıt (rejim-bağlı) |
| 19 | DINOv2 multilayer (taps=2) | **seg 0.60→0.29 (çöktü)** | ⚠️ **gerçek DPT değil, kaba proxy** (füzyonsuz + erken katman); "donuk öldürür" geri çekildi |
| 20 | **ResNet + PAN** (det neck) | **det 0.19→0.205 (+%6.4)** 🎯 | det zayıflığı da kısmen **NECK'miş** (ASPP'nin det ikizi); small AP 0.064; conv omurga task-neck'ten faydalanır |
| 21 | DINOv2 + CIoU (box loss) | det düz, AP@0.75 sıçramadı | **loss da det'i çözmedi** → DINOv2 det **FEATURE-bound** |
| 22 | Tek-görev ablasyonu (det/seg/cls-only) | det≈, seg≈, **cls +%4** | **ASİMETRİK interference:** cls eziliyor, det/seg değil |
| 23 | Referans modeller | — | frozen DINOv2, bizim-veride-trained specialist'leri geçiyor (aşağıda) |

**Deneme 19 mekanizması (multilayer neden çöktü):** (1) ViT'te derinlik=semantik-olgunluk, çözünürlük değil
(CNN "erken=detay" sezgisi çürür); (2) DINOv2 semantiği son katmanda; (3) füzyonsuz-partition seg'i son-katman
semantiğinden kopardı + yanlış katman (blok 5 çok erken); (4) cls sabit kaldı = son-katmanı okuyan tek head =
mekanizma kanıtı. Gerçek DPT (son-N + füzyon) donuk DINOv2'de çalışır → future work.

**Deneme 22 mekanizması (asimetrik interference):** cls global feature (level-3+GAP) ister; det/seg uzamsal+
yoğun → paylaşılan neck'e çok gradyan akıtıp cls'i dışlıyor. → detection **interference-immune/feature-bound**;
classification **interference-bound (headroom var)**. D16 (adaptif loss cls'i +3-4% besledi) ile tutarlı.

### Deneme 23 — Referans modeller (görev-başı, multi-task'ımızla kıyas)
| görev | referans | rejim | referans | bizim multi-task |
|---|---|---|---|---|
| det | ResNet50+RetinaNet | trained, bizim 22.5k, tek-görev | 0.1932 | DINOv2 0.2300 · ResNet 0.1965 |
| det | Faster R-CNN | zero-shot, full-COCO (SOTA tavanı) | **0.4690** | 0.2300 |
| seg | ResNet50+ASPP | trained (backbone açık), tek-görev | 0.3751 (step48k kesildi) | 0.6011 |
| seg | **SegFormer-B2** | trained, bizim 22.5k, tek-görev | **0.5109** | 0.6011 |
| cls | ResNet50 classifier | trained, bizim 22.5k, tek-görev | **0.6857 / 0.6767** | 0.7800 / 0.7239 |

**🎯 Referans bulguları:**
- **Trained ResNet det (0.19) ≈ frozen ResNet, frozen DINOv2'nin (0.23) altında.**
- **Faster R-CNN zero-shot 0.47 = SOTA tavanı** (~2× bizim); asıl gap **küçük nesnede** (SOTA small 0.310 vs
  bizim 0.036, ~9×) → DINOv2 det açığının çekirdeği **çözünürlük/ince-feature** (patch14 kaba; ViTDet dersi).
- **Trained ResNet seg (0.375) < frozen ResNet+ASPP (0.46)** → backbone'u 22.5k'da fine-tune etmek seg'i
  **kötüleştirdi** (drift/overfit; donuk daha sağlam) → frozen-foundation tezini pekiştiriyor.
- **🎯🎯 SegFormer-B2 (0.51) < frozen DINOv2 (0.60)** → **modern+trained+bizim-veride bir segmenter bile frozen
  DINOv2'yi geçemedi.** Tezin en güçlü tek kanıtı. Sıralama: DINOv2 > SegFormer > ResNet+ASPP > trainable ResNet
  > ResNet+FCN. ⚠️ boyut confound (B2 27M < ViT-B 86M; B5 size-match test edilmedi).

---

## 3. Ana bulgular (sentez — tezin çekirdeği)

1. **Pretraining → downstream öngörüsü doğrulandı.** DINOv2 lider; her paradigma görev-imzalı. Kalite+tür
   belirleyici (paradigma değil): DINOv1 SSL zayıf, DINOv2 SSL en iyi.
2. **Adaptasyon yasası:** kazanç ∝ 1/donuk-kalite (MAE +%82 ↔ DINOv2 +%0.4). Pretraining "hangi adaptasyon
   rejiminde iyi"yi de öngörüyor.
3. **Detection FEATURE-bound (dört bağımsız negatif):** neck (task-native ✗, multilayer ✗), loss (CIoU ✗),
   interference (tek-görev ✗) → hiçbiri det'i oynatmadı; sadece **adaptasyon (LoRA +%16 ✓)**. ViTDet "det
   fine-tune ister" dersinin çok-yönlü doğrulaması. Açığın çekirdeği **küçük-nesne/çözünürlük** (Faster R-CNN
   referansı kanıtladı).
4. **Asimetrik interference:** classification multi-task'ta eziliyor (izole +%4), detection/seg değil (feature-bound).
5. **Decoder mediation (conv'a özel):** ResNet seg zayıflığı decoder'dı (ASPP +%43), det zayıflığı kısmen
   neck'ti (PAN +%6.4) — **conv omurga task-neck'ten faydalanır**; donuk ViT (DINOv2, tavan) tersine nötr/zararlı.
6. **Frozen foundation > trained specialist (sınırlı veride):** trained ResNet det/seg ve **modern SegFormer**
   bile frozen DINOv2 multi-task'ı geçemedi. + verimlilik: 1 paylaşılan backbone < 3 ayrı specialist (3× maliyet).
   → **kısıtlı platform için frozen-multitask kazanıyor** (motivasyon doğrulandı).

---

## 4. Bu sohbette yapılanlar (kod + deney + düzeltme)

### Kodlanan (commit'lendi)
- **`neck_mode` enum** (`shared | per_task_identical | task_native`) — `config.py` + `multitask_model.py`.
  task_native: det=SFP, seg=ASPP (ham trunk), cls=GAP (ham trunk). (commit c269321)
- **PAN detection neck** (`pan.py`, `det_neck: fpn|pan`) — bottom-up yol, yalnız detection'a. (1848588)
- **DINOv2 multilayer** (`multilayer_taps`) — `get_intermediate_layers` + kanal-concat (cache formatı bozulmadan). (1848588)
- **CIoU box loss** (`det_box_loss: l1|smooth_l1|giou|ciou`) — torchvision `_box_loss`, RetinaNet regresyon head `_loss_type`. (29289f9)
- **Tek-görev configleri** (`dinov2_{detonly,segonly,clsonly}`) — loss ağırlıkları sıfırlama, sıfır kod. (8fdd5ab)
- **"Normal model" referans configleri** (`resnet_{det,seg,cls}_ft` — backbone açık + tek-görev). (5736f45)
- **Referans scriptleri:** `eval_pretrained_detector.py` (zero-shot torchvision detektör + FPS/bellek),
  `train_segformer.py` (HF SegFormer fine-tune + bizim mIoU parite + `--eval-only`), `train_ref_segformer.yaml`.
  (16cea5a, db28225)
- **`requirements-gpu.txt`:** transformers<4.45 (torch 2.3.1 uyumu; yeni transformers torch≥2.4 istiyordu). (6321b3d)
- Testler: `test_neck_modes`, `test_pan_multilayer`, `test_smoke_forward`'a ASPP+ResNet.

### Koşulan deneyler (bu sohbet)
Deneme 17 (task-native), 18 (ResNet+ASPP +%43), 19 (multilayer çöktü), 20 (ResNet+PAN +%6.4), 21 (CIoU düz),
22 (tek-görev interference), 23 (referanslar: trained ResNet det/seg, Faster R-CNN zero-shot, SegFormer, cls koşuluyor).

### Kayıt bütünlüğü düzeltmeleri (dürüstlük)
- **DPT-proxy düzeltmesi:** "donuk öldürür" geri çekildi → asıl sebep füzyonsuz-partition + yanlış katman.
- **Tutarlılık denetimi:** (1) "DINOv2 tavanda" seg/cls'e özel, det'te headroom var (LoRA +%16); (2) multilayer
  artık "neck-invariance" kanıtı değil (proxy-confounded); (3) YOLOP "ekstra bağlam" vs ResNet+ASPP baseline-farkı.

### Kavramsal netleştirmeler (bu sohbet)
PSPNet (PPM) · YOLOP loss · DPT (neden proxy çöktü) · "baseline tek katman kullanıyor (5 seviye ≠ 5 katman)" ·
"ASPP bir neck'tir, head değil" · ViTDet (head değil, reçete; SFP'sini zaten kullanıyoruz) · MoE detection'ı
çözmez (interference darboğaz değil) · referans rejim-tutarlılığı · deployment-motivasyonu (küçük model şart).

---

## 5. Düşünülen ama UYGULANMAYAN fikirler

| fikir | neden yapılmadı / durum |
|---|---|
| **MoE / task-specific gate** (ROADMAP Faz 3 C) | Deneme 17+22 gösterdi: det interference-bound değil → MoE det'i çözmez; kod eskizi var, koşulmadı |
| **Instance segmentation** (Mask R-CNN, v2) | SAM bilmecesini çözer ("seg-native instance'da parlar") ama ağır mimari iş; `maskrcnn_v2.py` stub; future work |
| **YOLO seg/cls için** | görev uyuşmuyor (YOLO-seg=instance, YOLO-cls=single-label; bizimki semantic/multi-label) |
| **ASL / ML-Decoder** (COCO-multilabel cls) | fiddly (inplace-abn + dış repo + sınıf-eşleme); payoff küçük (cls zaten güçlü); atlandı |
| **Panoptic zero-shot seg** (Mask2Former→semantic) | 133→81 sınıf eşlemesi hataya açık; SegFormer daha güvenli → SegFormer seçildi |
| **CLIP zero-shot cls** | motivasyona uymuyor (büyük model, deployment kıyası değil); "accuracy bağlamı" olarak kalabilir |
| **Gerçek DPT (son-N + füzyon)** | multilayer proxy çöktükten sonra; ağır + DINOv2 seg tavanda → future work |
| **LoRA yasası mid-point** (DINOv1/CLIP+LoRA) | 2-nokta çizgiyi eğriye çevirir ama yeni precompute + uncached = pahalı; future work |
| **Full fine-tune** (trainable>0 ViT) | LoRA zaten "adaptasyon kaldıraç" mesajını verdi; marjinal ek; opsiyonel |
| **DINOv2 register/patch ablasyonu** | "DINOv2 neden en iyi" (SSL kalitesi mi ince-patch/register mi) — ROADMAP'te, ucuz ama tezin merkezi değil |
| **SAM pre-neck 768 tap** | 256-kanal confound'unu kaldırır; notlu caveat yeterli sayıldı |
| **FCOS (anchor-free head)** | "sıralama head'e bağlı mı" robustluk; orta değer, koşulmadı |
| **BEiT renorm re-run** | normalizasyon fix'i belli (0.5/0.5), koşulmadı — Faz 1'de açık delik |
| **per_task_identical (B) koşusu** | saf interference probu; task-native (D) ~düz çıkınca düşük öncelik |
| **det-upweight loss** | Deneme 16 det'in aç kaldığını gösterdi; elle upweight ucuz ama koşulmadı |
| **LR scheduler** | şu an sabit lr; det yavaş yakınsıyor → cosine+warmup ucuz, keşfedilmedi |
| **CLIP zero-shot / Faster-R-CNN-türevi cls** | detektörü cls'e koşmak "multi-task gibi" olur + gerçek classifier değil → reddedildi |

---

## 6. Kod haritası (ne var)

- **Backbone'lar** (`src/mtl/models/`): resnet(FPN), dino, dinov2(+multilayer), clip, mae, deit, moco, beit,
  sam, ijepa + `sfp.py` (paylaşılan Simple Feature Pyramid), `pan.py` (PAN neck), `lora.py`.
- **Head'ler:** detection (RetinaNet), segmentation (FCN/ASPP/LR-ASPP), classification (GAP+FC). `multitask_model.py` = kalp.
- **Loss:** `joint_loss.py` (sabit ağırlık + Kendall `UncertaintyWeighter`).
- **Config alanları (Faz 3):** `seg_neck` (fcn/aspp/lraspp), `neck_mode`, `det_neck` (fpn/pan), `multilayer_taps`,
  `det_box_loss` (l1/ciou), `adaptive`, LoRA (`lora*`).
- **Scriptler:** train, train_cached, eval, precompute_features, benchmark_latency, visualize, plot_metric_curves,
  infer_video(_grid), compare_results, prepare_coco_subset, download_subset_images, **eval_pretrained_detector**
  (yeni), **train_segformer** (yeni).
- **Configler:** her backbone + Faz 3 varyantları (adaptive, taskneck, taskneck_native, segaspp, seglraspp,
  multilayer, ciou, detonly/segonly/clsonly, resnet_*_ft, ref_segformer).

---

## 7. Kalan işler / gelecek

**Tier 1 — tezi tamamlayan (ucuz, kritik):**
- **Held-out TEST seti** final sayıları (Faz 2'de seçim yaptık → val'e overfit değil kanıtı). Saf eval, ~bedava. (`instances_test_subset.json` hazır.)
- **Rapor figürleri:** kalitatif viz (section 12) + öğrenme eğrileri (section 13) + **doğruluk↔FPS scatter** (motivasyon).
- **Yazım / konsolidasyon** (RESULTS zaten %70'i).

**Tier 2 — güçlendiren:** SegFormer verim ölçümü · BEiT renorm re-run · per_task_identical (B) · cls-ft/SegFormer nihai epoch.

**Yayın için (ana venue):** merkezi iddiayı keskinleştir ("**donuk foundation, sınırlı-veri+kısıtlı-platformda
trained specialist'leri geçer**") + multiple seeds/error bar + (opsiyonel) küçük method katkısı.

---

## 8. Referanslar (cite edilecekler)

- **ViTDet** (Li et al. 2022) — Simple Feature Pyramid + "plain ViT detection fine-tune ister" (bizim neck + det dersi).
- **DPT** (Ranftl 2021) — multi-layer dense fusion (multilayer'ın gerçek hali).
- **SETR-MLA** (Zheng 2021) — multi-level aggregation.
- **DeepLab/ASPP** (Chen) — seg bağlam modülü (ResNet+ASPP kazancı).
- **PSPNet/PPM** (Zhao 2017) — bağlam-toplama seg soyağacı.
- **YOLOP** (2022) — multi-task + PAN + CIoU (det neck/loss).
- **PANet** (Liu 2018) — bottom-up path (PAN).
- **CIoU** (Zheng 2020) — Complete-IoU box loss.
- **Kendall et al. 2018** — uncertainty weighting (adaptif loss).
- **SegFormer** (Xie 2021), **CLIP**, **DINO/DINOv2**, **MAE**, **SAM**, **I-JEPA**, **DeiT**, **BEiT**, **MoCo v3**, **LoRA** (Hu 2021).

---

*Son güncelleme: 2026-08-06. Deney günlüğü: EXPERIMENTS.md (Deneme 1–23). Sonuç tabloları: RESULTS.md. Bu dosya
tüm oturumların/kodun/MD'lerin sentezidir. Referans seti tamam (det/seg/cls trained + Faster R-CNN/SegFormer);
bazı trained referanslar loss-platoda erken kesildi (nihai epoch opsiyonel).*
