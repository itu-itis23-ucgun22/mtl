# Deney Yol Haritası — Foundation Model × Multi-Task Karşılaştırması

Bu dosya **hangi deneyi hangi sırada** koşacağımızı ve **neyi dahil edip neyi parkladığımızı**
belirler. Amaç: kombinatoryal patlamaya boğulmadan, tek-değişken disipliniyle savunulabilir
bulgular çıkarmak. Sonuçlar: [RESULTS.md](RESULTS.md). Anlatı/gerekçe: [EXPERIMENTS.md](EXPERIMENTS.md).

## Tez (tek cümle)
> **Pretraining sinyali (foundation model), az-etiketli çok-görevli dense tahminde hangi
> downstream görevde iyi olacağını öngörür mü?** (supervised vs SSL vs dil vs segmentation-native)

## Altın kurallar
1. **Grid değil, tek eksen.** Her deney, sabit bir kanonik protokolden **tek** şeyi değiştirir.
   Yoksa confound'a boğulursun (yaşadık: 16 vs 1.6 epoch, batch farkı).
2. **Protokolü kilitle** (Faz 0), sonra eksenleri tek tek gez.
3. **Backbone'ları paradigma çeşitliliğiyle seç** — 5 temsilci > 10 benzer model.
4. **Odak > kapsam.** Yeni parlak fikir (yeni MTL mimarisi, yeni head...) gelince "bu tezde mi,
   sonraki çalışmada mı" diye sor. Çoğu → parkla.

## ⭐ Varsayılan iş akışı: FEATURE CACHING (donuk koşularda hep)
**Bundan sonraki tüm DONUK-backbone deneylerinde feature-caching kullan.** Donuk backbone
çıktısı deterministik olduğu için bir kez hesaplanıp diske yazılır; sonra sadece neck+head
cache'den eğitilir → pahalı ViT forward'ı her adımda tekrar koşulmaz, sweep ucuzlar.

```
# 1) trunk feature'larını bir kez yaz (donuk backbone; DINO ailesi)
python scripts/precompute_features.py --config <cfg> --split train
python scripts/precompute_features.py --config <cfg> --split val     # (eval istersen)
# 2) neck+head'i cache'den eğit (ViT forward YOK)
python scripts/train_cached.py --config <cfg>
# 3) normal eval (checkpoint tam model state'i içerir, uyumlu)
python scripts/eval.py --config <cfg> --checkpoint checkpoints/<run>_cached_epoch15.pt
```
Kod: `trunk_forward`/`neck_forward` (dino[v2]_backbone.py), `MultiTaskModel.forward_from_trunk`,
`scripts/precompute_features.py`, `datasets/cached_features.py`, `scripts/train_cached.py`.

**Kısıtlar (bilerek):** yalnız DONUK backbone (çözükse cache geçersiz); augmentation yok (cache
flip'siz); DINO ailesi (`trunk_forward`'ı olanlar — ResNet donuk gövdesi zaten ucuz, cache gereksiz).
**Depolama:** ViT-B trunk ~1.5-2 MB/görsel (float16) → 22.5k ≈ 35-45 GB/backbone. Bir backbone'u
cache'le → deneylerini koş → cache'i sil → sıradakine geç. Prototip için küçük subset (n_images).

## Durum (2026-07-07)
- ✅ **İlk bulgu** (RESULTS.md): donuk ResNet vs donuk DINO, 16 epoch, batch 4 → ResNet det/cls'de,
  **DINO seg'de** önde. "Göreve göre backbone tercihi."
- ✅ DINOv2 backbone eklendi (`dinov2_backbone.py`), henüz koşulmadı.
- ✅ Feature-caching altyapısı eklendi (bu dosyadaki iş akışı).

---

## Faz 0: Kanonik protokolü kilitle
Sabitler: **multi-task, RetinaNet, sabit loss ağırlıkları, DONUK backbone, 16 epoch, batch 4,
img 512 (patch14 backbone'larda 518), seed 42.** Her deney bundan tek şey değiştirir.
- [ ] **Uncertainty (Kendall) loss** kararı: baseline'da (DINOv2 ya da ResNet) bir kez test et.
      Yardım ediyorsa **tüm sweep için kanonik yap ve KİLİTLE** (sweep ortasında loss değiştirme →
      tüm kıyaslar bozulur). `joint_loss.py`'de v2 olarak not düşülü.

## Faz 1: Backbone sweep — ÇEKİRDEK (donuk, cache'li, 16 epoch)
Sabit protokol altında her foundation model. Paradigma temsilcileri:

| # | Paradigma | Model | Durum |
|---|---|---|---|
| 1 | Supervised classification | resnet50 (ImageNet) | ✅ var (RESULTS satır 8) |
| 2 | SSL distillation | dinov2_vitb14_reg | ⏳ sıradaki |
| 3 | Image-text (dil) | clip_vitb16 (CLIP ViT-B/16, OpenAI) | ✅ koşuldu (Deneme 8): det 0.142 / seg 0.440 / cls 0.690/0.650 |
| 4 | Segmentation-native | SAM (image encoder) | ☐ backbone wrapper gerek |
| (5)| Predictive SSL | I-JEPA | ☐ opsiyonel |
| (-)| SSL distillation v1 | dino_vitb16 | ✅ var (baseline) |

- [ ] DINOv2 sweep koşusu (cache + train_cached + eval).
- [x] CLIP wrapper (`clip_backbone.py`, DINO desenini izler; patch16 → DINOv1 ile aynı grid;
      norm backbone içinde ImageNet→CLIP). [x] CLIP sweep koşuldu (Deneme 8): CLIP vs DINOv1 en temiz
      kıyas → dil-pretraining semantik-güçlü/lokalizasyon-zayıf; DINOv2 hâlâ dört metrikte lider.
- [ ] SAM wrapper (`sam_backbone.py`) → sweep.
- Not: DINOv1/v2/v3 hepsini koşma (aynı aile, tekrar). Birini temsilci al; ilerlemeyi istersen
  küçük bir alt-çalışma olarak göster.

## Faz 2: Adaptasyon ekseni (donuk → LoRA → full) — 1-2 backbone
Tüm backbone'larda değil; temsilcide (DINOv2 + ResNet). Test: fine-tune sıralamayı değiştiriyor
mu / detection açığını kapatıyor mu (ViTDet: ViT detection'da fine-tune ister).
- [ ] LoRA implementasyonu (`trainable_blocks`'a alternatif, PEFT). **Not:** LoRA'da backbone
      çözülür → **cache KULLANILAMAZ** (trunk her adım değişir), normal `train.py` ile koş.
- [ ] full fine-tune (`trainable_backbone_layers>0`) — VRAM'e göre.

## Faz 3 (opsiyonel, rigor/ekstra)
- [ ] **Tek-görev ablasyonu** (1-2 backbone): sadece det / sadece seg / sadece cls → multi-task
      ile kıyasla. Task interference'ı ölç; detection açığı multi-task'tan mı backbone'dan mı ayır.
- [ ] **Detection head ekseni** (FCOS, anchor-free): sıralama head'e bağlı mı? (`det_head` config'i)
- [ ] AP@0.50 vs AP@0.75 analizi: DINO detection açığı lokalizasyon mu sınıflandırma mı.

### Geriye dönük ablasyonlar — "DINOv2 neden en iyi?" (opsiyonel, belki deneriz)
Deneme 7: donuk DINOv2 dört metrikte de en iyi (seg 0.60 açık ara). Ama DINOv2, DINOv1'den **üç**
şeyde farklı (RESULTS.md "Yorum nüansı"): (a) daha iyi SSL (LVD-142M), (b) patch14+518 → ince grid
37×37, (c) register token'lar. Üstünlüğün ne kadarı **pretraining kalitesi** ne kadarı **mimari
(ince patch / register)** — bunu ayırmak için:
- [ ] **Register ablasyonu (EN UCUZ):** `dinov2_vitb14` (reg'siz) vs `dinov2_vitb14_reg`. Tek fark
      register → patch/pretraining aynı. Ekstra maliyet ~yok. **İlk denenecek ablasyon.**
- [ ] **Patch ablasyonu (DINOv1 içinde):** ViT-B/16 vs **ViT-B/8** (`vit_base_patch8_224.dino`,
      `PATCH_SIZE=8`). Aynı DINOv1 pretraining, tek fark patch → "ince patch ne kadar yarıyor".
      **PAHALI:** patch8@512 → 64×64 = 4096 token, attention ~16× (VRAM/hesap ağır); donuk olduğu
      için feature-caching yardımcı ama precompute yine yavaş.
- [ ] **Grid eşitleme:** DINOv1 patch16'yı 592px besle → 37×37 (DINOv2 grid'iyle eşleşir), pretraining
      farkını grid sabitken kıyasla (patch boyutu 14 vs 16 yine farklı kalır).
- Not: bunlar **tezin merkezi değil** — "DINOv2 en iyi" sonucu zaten net; sadece *neden*'i inceltir.

## Parklanan (bu tezin ekseni DEĞİL — sonraki çalışma)
- **MTL mimarileri** (Cross-Stitch, Sluice, Deep Relationship, Fully-Adaptive): soft sharing =
  backbone'un birden çok kopyası → foundation-model çağına pahalı/uyumsuz. Hard sharing'de kal.
- DINOv3 / SAM2 / SAM3 gibi aynı-aile yeni sürümler: temsilci yeterli; hepsini koşma.

## Yeni foundation model eklerken (checklist)
1. `src/mtl/models/<name>_backbone.py` — DINO desenini izle (`trunk_forward`/`neck_forward`,
   `.out_channels=256`, 5-seviye çıktı sözleşmesi). Donuk gövde + Simple Feature Pyramid.
2. `build_backbone` dispatch (`backbone.py`) + config (`configs/train_colab_<name>.yaml`).
3. Test (`tests/test_<name>_backbone.py`), ARCHITECTURE.md tablosu, notebook hücresi.
4. **precompute_features → train_cached → eval** (donuk sweep akışı).
5. Sonucu RESULTS.md ana tabloya + EXPERIMENTS.md'ye işle (tek değişken backbone diye niteleyerek).
