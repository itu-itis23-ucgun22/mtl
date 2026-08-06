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
| 12 | sam_vitb16  | 0                | ~90000 | 0.1497      | 0.1933   | 0.3410  | 0.3545 | **16 epoch donuk — SAM (Deneme 10)**; batch 4, img 512 (pos-embed/rel-pos interp; trunk 256-kanal). **SEG'DE EN DÜŞÜK** — sınıf-agnostik pretraining ≠ semantik seg (bkz. bulgu). Confound: 256-ch neck darboğazı |
| 13 | ijepa_vith16 | 0               | ~90000 | 0.1941      | 0.4264   | 0.6149  | 0.6047 | **16 epoch donuk — I-JEPA (Deneme 11)**; ⚠️ **ÇİFT CONFOUND → DİPNOT**: (a) ViT-H (632M vs ViT-B 86M), (b) neck sıkıştırması 1280→256=**5×** vs ViT-B 768→256=3×. batch 4, img 512. AP@0.50=0.357, small AP 0.057. Bkz. dipnot bulgusu |
| 14 | deit_vitb16 | 0                | ~90000 | 0.1369      | 0.4271   | 0.6871  | 0.6513 | **16 epoch donuk — DeiT (Deneme 12)**; batch 4, img 512 (patch16, aynı grid). **Supervised ViT → adil çekirdeğin supervised ayağı.** cls'de güçlü (~CLIP); DINOv1'i (SSL) cls/seg'de geçer. AP@0.50=0.278 |
| 16 | mae_vitb16 | **LoRA** | ~90000 | **0.2394** | **0.4640** | **0.6388** | **0.6180** | **🎯 FAZ 2 — MAE+LoRA (Deneme 15)**; donuk MAE (satır 11) ile AYNI 22.5k veri, tek fark LoRA (rank8, qkv+proj). Donuğa göre **det +%79 / seg +%82** → *"donukken kötü, çözülünce harika"* DOĞRULANDI. **Detection'da DINOv2-donuk'u geçiyor** (0.239>0.230). AP@0.50=0.399 |
| 18 | dinov2_vitb14_reg | 0 **(adaptif loss)** | ~90000 | 0.2155 | 0.5971 | **0.8053** | **0.7534** | **🔬 FAZ 3 — DINOv2 + adaptif loss (Deneme 16)**; donuk DINOv2 (satır 9) ile tek fark: sabit 1/1/1/0.5 yerine öğrenilen Kendall ağırlıkları. Öğrenilen exp(-s): cls **29**≫seg 5.9>det 3.6>bbox 2.5. **Kazanç değil yeniden-dağıtım:** cls +%3-4, det −%6, seg sabit → küçük-magnitüdlü cls'i besledi, zor detection'ı aç bıraktı |
| 17 | dinov2_vitb14_reg | **LoRA** | ~90000 | **0.2662** | **0.6036** | **0.8156** | **0.7787** | **🎯 FAZ 2 — DINOv2+LoRA (Deneme 15)**; donuk DINOv2 (satır 9) ile aynı veri. Donuğa göre **det +%16 / seg +%0.4 (tavan) / cls +%5-8** → zayıf MAE (+%80) ile TERS: **adaptasyon kazancı donuk-kaliteyle ters orantılı**. LoRA'da da dört metrikte lider (tam flip değil). AP@0.50=0.445 |
| 19 | dinov2_vitb14_reg | 0 **(task-native neck)** | ~90000 | 0.2277 | 0.5951 | 0.7947 | 0.7318 | **🔬 FAZ 3 — DINOv2 task-native neck (Deneme 17)**; donuk DINOv2 (satır 9) ile tek eksen: paylaşılan SFP yerine **det=SFP / seg=ASPP / cls=GAP** (native, ham trunk). Baseline'a ~eşit: **cls +%1.9, det/seg −%1** → **neck oynatmak yardım etmedi** (ROADMAP tahmini). ⚠️ "tavan" **seg/cls'e özel** (LoRA seg +%0.4); **det'te headroom VAR** (LoRA det +%16) ama neck det kaldıracı değil (ViTDet: fine-tune). Alt-bulgu: donuk DINOv2 seg'inde **çözünürlük > bağlam**. ⚠️ çok-değişkenli (seg decoder da SFP+FCN→ASPP değişti) → saf interference DEĞİL; `per_task_identical` (B) sırada |
| 20 | resnet50 | 0 **(seg=ASPP)** | ~90000 | 0.1923 | **0.4606** | 0.7081 | 0.6882 | **🎯 FAZ 3 — donuk ResNet + ASPP seg decoder (Deneme 18)**; donuk ResNet+FCN (satır 8) ile tek fark `seg_neck=aspp`. **seg 0.3215→0.4606 = +%43** (det/cls sabit) → ResNet seg zayıflığı **DECODER'mış, feature değil.** DINOv2+ASPP (satır 19, tavan→0) ile TAM ZIT: ASPP conv'da (DeepLab eşleşmesi) uçuruyor, ViT'te sıfır. Seg'de **2.'ye** fırlıyor (yalnız DINOv2 üstünde; ⚠️ diğerleri FCN → confound). AP large=0.298 |
| 21 | dinov2_vitb14_reg | 0 **(multilayer taps=2)** | ~90000 | 0.2031 | **0.2882** | 0.7787 | 0.7320 | **🔬 FAZ 3 — DINOv2 çok-katmanlı aggregation (Deneme 19)**; baseline (satır 9) ile tek fark: son katman yerine 2 derinlikten feature (blok 5+11), seviyeler derinliğe göre. **ZARAR: seg 0.601→0.288 (−%52 ÇÖKTÜ), det −%12, small AP 0.036→0.012.** cls sabit (0.78) — çünkü tek "son katmanı" okuyan head oydu (kanıt). ⚠️ **Gerçek DPT DEĞİL, kaba proxy:** asıl sebep füzyonsuz-partition (seg'in seviyesi son katmandan koptu) + yanlış katman (blok 5 çok erken); "donuk öldürür" DÜZELTİLDİ — donuk DINOv2+gerçek-DPT çalışır. Ders: ViT'te erken katman detay değil ham-semantik (CNN sezgisi çürür); ViTDet "son katman" tercihini dolaylı doğrular. Detay: EXPERIMENTS Deneme 19. |
| 22 | resnet50 | 0 **(seg=ASPP + det=PAN)** | ~90000 | **0.2047** | 0.4664 | 0.7063 | 0.6892 | **🎯 FAZ 3 — donuk ResNet + ASPP(seg) + PAN(det) (Deneme 20)**; satır 20 (aspp+**fpn**) ile tek fark det_neck=pan. **det 0.1923→0.2047 = +%6.4** (seg/cls sabit → PAN yalnız det yoluna). small AP **0.064** (bottom-up lokalizasyon → küçük nesne). → **det zayıflığı da kısmen NECK'miş** (ASPP'nin det ikizi); ASPP kadar büyük değil çünkü det zaten FPN çok-ölçeğini kullanıyordu (artımlı). **Conv omurga task-neck'ten faydalanır** (DINOv2 tersine). Verim: 27M/187 FPS. AP@0.50=0.352 |
| 23 | dinov2_vitb14_reg | 0 **(CIoU box loss)** | ~90000 | 0.2289 | 0.6005 | 0.7814 | 0.7367 | **🔬 FAZ 3 — DINOv2 + CIoU kutu loss (Deneme 21)**; baseline (satır 9, L1) ile tek fark: L1→CIoU (örtüşme+merkez+en-boy). **det 0.2300→0.2289 (düz), AP@0.75=0.225 sıçramadı** → CIoU'nun vaadi (lokalizasyon) gerçekleşmedi. **Loss da det'i çözmedi** → DINOv2 det **feature-bound** (neck ✗, loss ✗, sadece LoRA +%16 ✓). ⚠️ CIoU fine-tuned'da yerleşik; null **donuk-rejime özel**. cls_F1 +1.8% dolaylı (paylaşılan neck). AP@0.50=0.410 |
| — | beit_vitb16 | 0                | ~90000 | (geçersiz)  | (geçersiz)| (geçersiz)| (geçersiz)| **BEiT (Deneme 13) — ANA TABLOYA HENÜZ ALINMADI.** 224 ve 512'de çok düşük çıktı (seg ~0.06) ama sebep **normalizasyon uyuşmazlığı** (BEiT 0.5/0.5 ister, ImageNet-norm verdik) → renorm fix'i sonrası yeniden koşulacak. Bkz. EXPERIMENTS Deneme 13, GÜNCELLEME 2. |


> Deneme 2 (resnet, iddia edilen layers=0, ~200? adım): det 0.0013 / seg 0.022 / cls_mAP 0.128 /
> cls_F1 0.239 — **config doğrulanmadı, güvenilmez**, o yüzden ana tabloya gerçek bir "layers=0"
> satırı olarak alınmadı. Gerçek donuk-ResNet koşusu artık satır 6'dır.


## 🔬 Task-interference: tek-görev vs multi-task (donuk DINOv2 — Deneme 22)

Donuk DINOv2'yi her görev **tek başına** eğitip (diğer loss'lar 0) multi-task (satır 9) ile kıyas.
Her görevin **kendi** metriği geçerli (diğerleri eğitilmemiş → çöp). Tek fark = diğer görevlerin varlığı.

| görev | tek-görev | multi-task (satır 9) | delta | yorum |
|---|---|---|---|---|
| detection_mAP | 0.2231 | 0.2300 | **−3.0%** | interference YOK → det **FEATURE-bound** (dördüncü kanıt: neck✗/loss✗/interference✗/LoRA✓) |
| seg_mIoU | 0.5931 | 0.6011 | **−1.3%** | interference YOK (hafif pozitif transfer) |
| **cls_mAP** | **0.8117** | 0.7800 | **+4.1%** | **GERÇEK interference: multi-task cls'i eziyor** |
| **cls_F1** | **0.7573** | 0.7239 | **+4.6%** | " |

**Kazanım / gözlem:** interference **asimetrik**. det/seg (uzamsal, yoğun) paylaşılan neck'e çok gradyan akıtır
→ birbirini frenlemez, hatta hafif yardım. **cls (global, level-3+GAP) dışlanır** → tek başına +%4. Yani "tek
fazla görev" (görüntü-cls) iki yoğun görevin baskıladığı neck'te aç kalıyor (D16 adaptif-loss'un cls'i
up-weight'leyince +%3-4 vermesiyle tutarlı). **Rafine sonuç:** detection interference-immune/feature-bound;
classification interference-bound (headroom var). Detay: EXPERIMENTS Deneme 22.


## 📐 Referans modeller: multi-task modelimiz vs "normal / SOTA model" (görev-başı — Deneme 23)

Amaç: multi-task modelimizi **her görevde bir referansla** kıyasla (referanslar birbiriyle DEĞİL).
Her referans farklı **REJİM** → mutlaka etiketiyle oku. Metrik paritesi: hepsi bizim val + bizim metrik.

| görev | referans model | rejim | referans | bizim multi-task | gözlem |
|---|---|---|---|---|---|
| **det** | ResNet50+RetinaNet | trained, bizim 22.5k, tek-görev | **0.1932** | DINOv2 0.2300 · ResNet 0.1965 | trained ResNet ≈ frozen ResNet, **frozen DINOv2'nin ALTINDA** |
| **det** | Faster R-CNN | zero-shot, full-COCO (SOTA tavanı) | **0.4690** | DINOv2 0.2300 | SOTA **~2×** bizim; asıl gap **SMALL** (0.310 vs 0.036, ~9×) |
| **seg** | ResNet50+ASPP | trained (backbone AÇIK), bizim 22.5k, tek-görev | **0.3751** | 0.6011 | step 48000/90000'de **kesildi** (loss platoda, yakınsamış). trainable ResNet **frozen ASPP'nin (satır 20: 0.46) ALTINDA** → fine-tune 22.5k'da seg'e yaramadı |
| **seg** | SegFormer-B2 | trained, bizim 22.5k, tek-görev | **0.5109** | 0.6011 | modern trained segmenter → ResNet ref'lerin ÜSTÜNDE ama **frozen DINOv2'nin ALTINDA** (⚠️ B2 27M < ViT-B 86M, boyut confound) |
| **cls** | ResNet50 classifier | trained (backbone AÇIK), bizim 22.5k, tek-görev | **0.6857 / 0.6767** | 0.7800 / 0.7239 | step 58000 (loss ~platoda). trained ResNet **frozen ResNet multi-task'ın (0.7084) ALTINDA** → fine-tune 22.5k'da cls'e de yaramadı |

**🎯 Detection bulgusu (trained ResNet ref):** backbone açık + tek-görev trained ResNet detektör (0.1932,
loss platoda→yakınsamış) **frozen ResNet multi-task'a ≈ eşit** (0.1965) ve **frozen DINOv2'nin altında**
(0.2300). → Bu veri ölçeğinde (22.5k) **frozen foundation feature'ı, konvansiyonel-trained ResNet
detektörden daha iyi** — yani DINOv2'nin mutlak-düşük 0.23'ü "kötü" değil, normal trained ResNet'i geçiyor.
**Alt-gözlem:** trained ResNet small-obj AP **0.070** > DINOv2 **0.036** (ResNet+FPN @512 ince spatial;
DINOv2 patch14 kaba) — ama genel mAP'te DINOv2 önde. ⚠️ REFERANS (kıyas değil): rejim farkları etiketli.

**🎯 Detection SOTA tavanı (zero-shot Faster R-CNN, full-COCO):** **0.4690** (AP@0.50=0.677; small 0.310 /
med 0.525 / large 0.611) → frozen-multitask'ımızın (**0.2300**) **~2 katı.** Gap çok-faktörlü (full-veri 118k
vs 22.5k + trained backbone + tek-görev + 2-stage Faster R-CNN + yüksek çözünürlük 800-1333). **Asıl gap küçük
nesnelerde:** SOTA small **0.310** vs bizim **0.036** (**~9×**). → Frozen DINOv2 detection açığının çekirdeği
**küçük-nesne/çözünürlük**: patch14 kaba grid + 518 çözünürlük küçüğü kaçırıyor; Faster R-CNN yüksek-res + FPN +
RoIAlign ile parlıyor. Bu, "det kaldıracı = ince-feature/çözünürlük (fine-tune/ViTDet)" hikâyesinin en net kanıtı.
Bağlam: **0.23 "kötü" değil** — frozen + multi-task + 22.5k kurulumun tavana (full-optimize SOTA) uzaklığı bu kadar.

**🎯 Segmentation (trained ResNet ref):** trainable ResNet+ASPP seg-only (**0.3751**, train loss platoda→yakınsamış)
**frozen ResNet+ASPP'nin (satır 20: 0.4606) ALTINDA** ve frozen DINOv2'nin (0.6011) çok altında. → **Backbone'u
22.5k'da fine-tune etmek seg'i İYİLEŞTİRMEDİ, hatta frozen'a göre kötüleştirdi** — az veride güçlü pretrained
backbone'u **çözmek, dondurmaktan kötü** (backbone drift/overfit; frozen ImageNet feature'ı daha sağlam). Bu,
projenin **frozen-foundation tezini pekiştiriyor:** referans (tavan sanılan) bizim frozen yaklaşımımızın **altında**
çıktı. (⚠️ kısmen LR/optimizasyon olabilir ama loss yakınsamış → gerçek etki.) Detection ref'iyle aynı melodi:
trained ResNet ≈/< frozen, **frozen DINOv2 hepsinin üstünde.**

**🎯🎯 Segmentation (SegFormer ref — modern trained segmenter):** SegFormer-B2 (bizim veride fine-tuned, seg-only)
= **0.5109** → ResNet ref'lerinin üstünde (0.46 / 0.375) ama **frozen DINOv2 multi-task'ın (0.6011) ALTINDA.**
Yani **purpose-built + modern + bizim veride eğitilmiş bir segmenter bile frozen DINOv2'yi geçemedi** — frozen-
foundation tezinin en güçlü tek kanıtı. Seg sıralaması: **DINOv2 (0.60) > SegFormer-B2 (0.51) > ResNet+ASPP (0.46)
> trainable ResNet (0.375) > ResNet+FCN (0.32).** ⚠️ **Boyut confound'u:** SegFormer-B2 encoder ~27M vs DINOv2
ViT-B ~86M (3×) → size-matched SegFormer-B5 (~85M) farkı kapatabilir; ama bu boyutta bile SegFormer geçemedi +
verim açısından (27M ≈ ResNet) rekabetçi (accuracy↔size tatlı noktası).

**🎯🎯🎯 "3 ayrı trained ResNet specialist" üçlüsü tamam — DEPLOYMENT payoff:** det 0.1932 · seg 0.3751 ·
cls 0.6857 (hepsi backbone açık + tek-görev, bizim 22.5k). **Üçünde de: trained ResNet specialist ≤ frozen
ResNet multi-task (0.1965/0.4606/0.7084) << frozen DINOv2 multi-task (0.2300/0.6011/0.7800).** → **3 ayrı
trained specialist (3× backbone maliyeti), 1 frozen paylaşılan backbone'u (1× maliyet) HİÇBİR görevde geçemiyor**;
frozen DINOv2 hepsini eziyor. Motivasyon (kısıtlı platform) doğrulandı: **frozen-multitask kesin kazanan** (hem
maliyet hem doğruluk). Tekrarlayan mekanizma: 22.5k'da trainable ResNet backbone drift/overfit → frozen ImageNet
feature'ı daha sağlam (seg/cls'de frozen'ın bile altında).

**⚡ Referans verim (A100, aynı ölçüm = adil):**
- **det-ft** (ResNet backbone+FPN): **26.8M / 92.6 FPS / 273 MB** (latency 10.80 ms) → bizim A100 Verimlilik
  tablosu **ResNet satırıyla birebir tutarlı** (91.1 FPS/26.8M) ✅ → A100'de kıyas adil.
- **SegFormer-B2** (FULL model, A100): **27.4M / 55.6 FPS / 919 MB** (latency 17.97 ms; seg 0.5109 nihai/epoch15).
  ⚠️ full-model (encoder+decode head) → det-ft'nin backbone-only'siyle kapsam farklı. Gözlem: SegFormer 27.4M
  **FULL segmenter** ≈ ResNet backbone (26.8M) boyutunda + makul hızlı (55.6 FPS) ama **tek görev + seg 0.51**;
  DINOv2 (92M backbone, 45 FPS) **üç görevi tek backbone'la** yapıyor + seg'de daha iyi (0.60). → seg-only edge
  dağıtımı için SegFormer-B2 verimli bir nokta; **çok-görevli** kısıtlı platform için paylaşılan DINOv2 üstün.
- **Deployment maliyeti (somut):** 3 ayrı ResNet specialist = **3× 26.8M ≈ 80M** + 3 forward; 1 paylaşılan frozen
  backbone = **1× 26.8M** + 3 minik head ≈ 1 forward. → paylaşılan **~3× ucuz** (backbone = pahalı parça), üstelik
  doğrulukta da geçiyor. Motivasyon (kısıtlı platform) sayısal olarak da doğrulandı.


## ⚡ Verimlilik — FPS / gecikme / bellek (A100-SXM4-40GB, batch=1)

Motivasyon (kısıtlı platform / uçak): en pahalı parça omurga → doğruluğun yanına **maliyet**. Ölçülen:
`build_backbone` (backbone + SFP neck) ileri-geçişi, **tek görsel (batch=1 = gerçek-zaman)**. `params_M`
SFP neck dahildir (~+5M → saf ViT-B ~86M yerine ~91M). Kaynak: `scripts/benchmark_latency.py`.

| backbone | img | params (M) | gecikme (ms) | **FPS** | tepe bellek (MB) |
|---|---|---|---|---|---|
| resnet50            | 512 | **26.8** | 10.98 | **91.1** | **237** |
| deit_vitb16         | 512 | 91.4 | 15.98 | 62.6 | 447 |
| dino_vitb16         | 512 | 91.4 | 15.89 | 63.0 | 447 |
| clip_vitb16         | 512 | 91.4 | 16.18 | 61.8 | 447 |
| mae_vitb16          | 512 | 91.4 | 16.43 | 60.9 | 447 |
| dinov2_vitb14_reg   | 518 | 92.1 | 22.03 | 45.4 | 474 |
| sam_vitb16          | 512 | 90.1 | 27.64 | 36.2 | 560 |
| ijepa_vith14        | 512 | **642.3** | **107.27** | **9.3** | **2729** |

**Bulgular:**
1. **ViT-B/16 @512 dörtlüsü (DeiT/DINOv1/CLIP/MAE) maliyette BİREBİR AYNI** (~62 FPS, 447 MB) — aynı mimari
   → **pretraining "bedava eksen": aynı maliyet, en iyi transfer edeni seç.** Doğruluk farkları **sıfır ek
   maliyetle** geliyor; adil çekirdeğin maliyet-eşitliğini de doğrular.
2. **ResNet-50 = verimlilik kralı** (91 FPS, 27 M, 237 MB) — en hafif/hızlı → kısıtlı platformda neden hâlâ
   güçlü baseline olduğunu açıklar.
3. **DINOv2** küçük maliyet öder (45 FPS) — patch14@518 daha ince grid (37×37) → ama dört metrikte lider →
   **iyi takas** (doğruluk↔maliyet tatlı noktası).
4. **SAM** daha ağır (36 FPS, 560 MB) — vasat sonuç için pahalı.
5. **I-JEPA ViT-H = "ölçek vergisi"**: en yavaş (9.3 FPS ≈ ResNet'in 1/10'u), 642 M param (24×), 2.7 GB
   bellek (11×) — **VE en iyi değil** (DINOv2 ViT-B dört metrikte onu geçiyor). *"Büyük model hem yavaş, hem
   pahalı, hem daha iyi DEĞİL"* için kesin kanıt.

**Ana çıkarım (doğruluk↔maliyet):** tatlı nokta = **DINOv2** (en iyi doğruluk, makul ViT-B maliyeti);
en ucuz = **ResNet**; anti-örnek = **I-JEPA** (en pahalı, doğruluk lideri değil). Doğruluk↔FPS grafiğinde
DINOv2 sağ-üstte (hızlıca iyi), I-JEPA sağ-altta (yavaş+vasat).


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

## 🎯🎯🎯🎯🎯 BULGU — SAM: "segmentation-native" YANILTICI (Deneme 10) — Faz 1 tamamlandı

SAM ViT-B/16 (seg-native) eklendi → **Faz 1'in altı omurgası tamam**. Sürpriz sonuç: sweep'in
"segmentation-native" temsilcisi **segmentasyonda EN DÜŞÜK** (0.193, MAE'nin 0.255'inden bile aşağı),
ve **classification'da da en düşük**.

| metrik | DINOv2 | ResNet | CLIP | DINOv1 | MAE | **SAM** | SAM sırası |
|---|---|---|---|---|---|---|---|
| detection_mAP | **0.2300** | 0.1965 | 0.1420 | 0.1541 | 0.1336 | 0.1497 | 4/6 |
| seg_mIoU | **0.6011** | 0.3215 | 0.4398 | 0.3928 | 0.2545 | **0.1933** | **6/6** |
| cls_mAP | **0.7800** | 0.7084 | 0.6899 | 0.5565 | 0.4215 | **0.3410** | **6/6** |
| cls_F1 | **0.7239** | 0.6799 | 0.6501 | 0.5515 | 0.4181 | **0.3545** | **6/6** |

**Bulgu: "segmentation-native" etiketi yanıltıcı — SAM'ın segmentasyonu bizimkiyle FARKLI bir görev.**
SAM, prompt'a (nokta/kutu) karşılık **sınıf-agnostik maske** üretmeyi öğrenir; **kategori bilgisi hiç
öğrenmez** ("bu bir maske" evet, "bu bir kedi" hayır). Bizim seg'imiz ise **semantik** (piksel→sınıf).
SAM feature'ları **nesnelik/sınır** taşır ama **kategori semantiği taşımaz** → hem semantik-seg hem
classification'da çöker (İKİSİ de kategori bilgisi ister). Bu, "seg-native ⇒ seg'de iyi" sezgisini
**çürütür**: hangi *tür* segmentasyon olduğu belirleyici.

**Not (detection ilginç):** SAM detection'da (0.150) MAE ve CLIP'in üstünde — nesnelik/sınır bilgisi
kutu bulmaya bir miktar yarıyor; ama sınıflandırma zayıf olduğu için mAP sınırlı kalıyor.

### ⚠️ İki confound (dürüstlük — güçlü sonuç çıkarmadan önce ablasyon gerek)
SAM'ın düşüklüğünün tamamı "sınıf-agnostik pretraining"e atfedilemez; iki ek fark var:
1. **256-kanal darboğazı:** SAM'ın kendi neck'i trunk'ı 768→256'ya sıkıştırır → SFP'ye giren bilgi
   diğer ViT'lerden (768) **dar**. Skorun bir kısmı bundan olabilir.
   → **Ablasyon adayı:** SAM'ın neck'inden ÖNCEki ham 768-d ViT çıktısını tap'la, tekrar koş.
2. **pos-embed/rel-pos interpolasyonu:** native 1024 → 512'ye indirdik (adil grid için). Hafif
   bozulma olabilir. → 1024 native koşu bir kontrol olur (ağır).
Yine de mekanizma (kategori semantiği yokluğu) hem seg hem cls'nin **birlikte** düşmesini açıklıyor;
tek başına 256-darboğaz cls'yi bu kadar düşürmezdi.

## 🏁 FAZ 1 ÖZET — altı omurga, beş paradigma (hepsi donuk, 16 epoch, batch 4)

**Adil çekirdek (ViT-B/16 @512, 32×32 grid, tek değişken pretraining):** DINOv1 · MAE · CLIP · SAM.
**Bağlam:** ResNet (supervised, conv/FPN), DINOv2 (patch14/518, en güçlü SSL).

**Ana sonuçlar:**
1. **DINOv2 dört metrikte de lider** — güçlü genel-amaçlı SSL her yerde kazanıyor.
2. **"Uzman" pretraining'ler donuk rejimde zayıf:** dil (CLIP), seg-native (SAM), reconstruction (MAE)
   — her biri kendi amacına göre bir eksende iyi ama genel donuk-transferde genel SSL/supervised'ın gerisinde.
3. **Her uzmanın imzası görevle örtüşüyor:**
   - **CLIP** (dil): semantik güçlü (cls/seg iyi), **lokalizasyon zayıf** (det en düşüklerden).
   - **MAE** (reconstruction): **donukken kötü** (semantik yok), düşük-seviye detay güçlü (small-obj en yüksek).
   - **SAM** (seg-native): **kategori semantiği yok** → semantik-seg + cls en düşük; nesnelik det'e biraz yarıyor.
4. **Tez doğrulandı:** pretraining sinyali, hangi downstream görevde parlayacağını **öngörüyor** — ve
   "uzman" sinyaller dar, "genel" sinyaller (DINOv2) geniş transfer sağlıyor.

**Faz 2 tahmini (Deneme 9'dan):** çözüldüğünde sıralama değişmeli — MAE en çok, DINOv2 en az kazanmalı.

## 🏆 ADİL ÇEKİRDEK TAMAM — beş paradigma, hepsi ViT-B/16 @512, tek değişken PRETRAINING (Deneme 12 ile)

DeiT eklendi → sweep'in **supervised ayağı da adilleşti**. Artık beş paradigma **birebir aynı mimaride**
(ViT-B, patch16, 32×32 grid, ImageNet norm, SFP, batch 4, 16 epoch) — **tek değişken pretraining hedefi.**

| metrik | DeiT (supervised) | DINOv1 (SSL-distill) | MAE (MIM) | CLIP (dil) | SAM (seg-native) |
|---|---|---|---|---|---|
| detection_mAP | 0.1369 | **0.1541** | 0.1336 | 0.1420 | 0.1497 |
| seg_mIoU | 0.4271 | 0.3928 | 0.2545 | **0.4398** | 0.1933 |
| cls_mAP | 0.6871 | 0.5565 | 0.4215 | **0.6899** | 0.3410 |
| cls_F1 | 0.6513 | 0.5515 | 0.4181 | **0.6501** | 0.3545 |

**🎯 En temiz kıyas — DINOv1 (SSL) vs DeiT (supervised), aynı ViT:**
Supervised (DeiT), SSL-distillation'ı (DINOv1) **classification'da açık ara** (0.687 vs 0.557) **ve
segmentasyonda** (0.427 vs 0.393) geçiyor; DINOv1 yalnızca **detection'da hafif önde** (0.154 vs 0.137).
→ Önceki "ResNet supervised, cls/det'te güçlü" bulgusu **conv/FPN artefaktı DEĞİLmiş**: supervised ViT de
aynı tanıma-gücünü gösteriyor. Supervised sınıflandırma-pretraining'i, donuk rejimde **tanıma görevlerine
(cls) doğrudan transfer** oluyor.

**🎯 Paradigma kümeleri:** DeiT (supervised) ≈ CLIP (dil) — ikisi de cls ~0.69, seg ~0.43, det ~0.14.
İkisi de **"görüntüde ne var" için optimize** (biri etiketle, biri metinle) → **tanıma-güçlü/lokalizasyon-orta**
aynı imza. Buna karşı MAE (reconstruction) ve SAM (seg-native) tanıma-zayıf (semantik yok).

**⚠️ Ama DINOv2 (daha iyi SSL) hepsini geçiyor** (det 0.230 / seg 0.601 / cls 0.780). Yani "supervised >
SSL" sonucu **DINOv1'e özgü**; pretraining KALİTESİ artınca (DINOv2) SSL her yerde öne geçiyor — tıpkı
CLIP bölümünde dediğimiz gibi. Ana mesaj: **paradigma değil, pretraining kalitesi+türü belirleyici.**

**Not (detection deseni):** Tüm ViT-B'ler detection'da düşük kümede (0.13–0.15); yüksek olanlar DINOv2
(0.230, ince grid) ve ResNet (0.197, gerçek FPN). → Detection'daki fark büyük ölçüde **grid/neck** kaynaklı
(pretraining'den çok) — Faz 3 head/grid ablasyonu için işaret.

## 📎 DİPNOT — I-JEPA ViT-H (Deneme 11): boyut-avantajlı, ADİL ÇEKİRDEK DIŞI

I-JEPA (predictive/latent SSL) eklendi ama **ViT-H (632M)** — Meta ViT-B yayınlamadığı için (bkz. Faz 1
notu). 7× boyut confound'u düzeltilemez → **ana kıyasa değil, dipnota.** Sonuç (donuk, 16ep, batch 4):
det 0.194 / seg 0.426 / cls_mAP 0.615 / cls_F1 0.605.

**İki bulgu (boyut confound'una rağmen anlamlı):**
1. **7× büyük olmasına RAĞMEN I-JEPA, DINOv2'yi (ViT-B) dört metrikte de geçemiyor** (det 0.194<0.230,
   seg 0.426<0.601, cls 0.615<0.780). → **"Büyük model ≠ iyi feature"; pretraining kalitesi (DINOv2)
   ham boyutu yeniyor.** Bu, confound'un *aleyhine* güçlü bir kanıt: en büyük modelimiz bile en iyi
   ViT-B'yi geçemedi. Ölçek her şey olsaydı I-JEPA lider olurdu; olmadı.
2. **I-JEPA >> MAE her metrikte** (seg 0.426 vs 0.255; cls 0.615 vs 0.422). Aynı "maskeli tahmin"
   ailesinde **latent-uzayda tahmin (I-JEPA) piksel yeniden-kurmayı (MAE) açık ara geçiyor** — I-JEPA
   makalesinin ana iddiası ("pikselleri değil, temsilleri tahmin et"). ⚠️ **Ama boyut confound'lu**
   (ViT-H vs ViT-B) → temiz atfedilemez; yön I-JEPA'nın iddiasıyla tutarlı, kesin kanıt değil.

### ⚠️ İKİNCİ confound — sıkıştırma oranı (I-JEPA aleyhine)
I-JEPA sadece boyutça farklı değil, **neck'te daha sert sıkışıyor:** embed_dim 1280 → SFP 256 = **5×**
sıkıştırma; ViT-B'ler 768 → 256 = sadece **3×**. Yani I-JEPA'nın zengin 1280-dim feature'ları dar neck'te
**daha çok boğuluyor** olabilir → I-JEPA'nın *aleyhine* bir ek confound. Bu, "boyut ≠ kalite" iddiasını
**hafifçe zayıflatır** (I-JEPA'ya "belki geniş neck'le kazanırdı" mazereti verir). Karşı-argüman: neck 1×1
lateral konvolüsyonu **eğitiliyor** → 1280→256 için göreve en uygun projeksiyonu öğrenir, rastgele darboğaz
kadar kayıplı değil. **Tasarım gerilimi:** embed_dim farklıysa "neck genişliği sabit" (256) ile "sıkıştırma
oranı sabit" birlikte tutulamaz — adil çekirdek (hepsi 768→256, 3×) temiz olduğu için **I-JEPA/SAM dipnot**.
→ Temiz test: I-JEPA'ya geniş neck (1280→512) ya da ViT-B I-JEPA (yok). Faz 3 ablasyonu.

**Konum:** I-JEPA det/seg'de güçlü (üst-orta), cls'de orta. Genel: DINOv2 < I-JEPA değil — yani en
büyük+predictive-SSL bile genel-amaçlı iyi SSL'in (DINOv2) gerisinde. Raporda **"boyut-kontrollü değil"**
etiketiyle sun.

## Nasıl güncellenir
- Colab'da her `scripts/eval.py` koşusu `runs/results.csv`'ye satır ekler.
- `python scripts/compare_results.py` → markdown tablo basar (buraya yapıştırılabilir).
- Yeni sayı geldikçe yukarıdaki `_?_` hücreleri doldurulur.
