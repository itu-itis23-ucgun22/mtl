# MTL — Sistemin Tam Haritası

Bu dosya kod değil, **zihin haritası**: hangi klasör/dosya neden var, hangi kod hangisini çağırıyor, bir görüntü baştan sona sistemden nasıl geçiyor. Amaç README (nasıl çalıştırılır) veya plan dosyası (neden bu kararlar alındı) değil — sistemin **omurgasını** tek bakışta görebilmek.

---

## 1. Tek cümlede sistem

Bir görüntüyü **bir kere** ResNet50+FPN'den geçiriyoruz, çıkan feature piramidini **üç bağımsız kafaya (head)** paylaştırıyoruz, her kafa kendi görevinin (detection/segmentation/classification) loss'unu üretiyor, dördü ağırlıklı toplanıp **tek** bir loss ile backbone + üç kafa **aynı anda** güncelleniyor.

```
                         ┌──────────────┐
   görüntü ──────────►   │  ResNet50    │
   (B,3,H,W)             │   + FPN      │  ← backbone + neck (birlikte)
                         └──────┬───────┘
                                │  features = {"0","1","2","3"}  (P2..P5, 256 kanal)
              ┌─────────────────┼─────────────────┐
              │                 │                 │
        features["0"]     tüm seviyeler      features["3"]
        (P2, stride 4)     (P2..P5)          (P5, stride 32)
              │                 │                 │
              ▼                 ▼                 ▼
      SegmentationHead   DetectionHead        ClassificationHead
      (FCN, piksel)      (RetinaNet, kutu)    (GAP+FC, çok-etiketli)
              │                 │                 │
         seg_loss      classification_loss   cls_loss
                       + bbox_regression_loss
              └─────────────────┼─────────────────┘
                                 ▼
                    joint_loss.combine_losses()
                    (ağırlıklı toplam: det_cls·1 + det_box·1 + seg·1 + cls·0.5)
                                 ▼
                        total.backward() + optimizer.step()
```

**Omurga takılabilir (gövde + neck).** Yukarıdaki "ResNet50 + FPN" kutusu aslında iki parça:
**gövde** (feature çıkaran omurga) + **neck** (gövde feature'larını head'lerin beklediği ortak
formata çeviren ara katman). FPN, ResNet'e özgü değil — sadece bir neck; torchvision
`resnet_fpn_backbone` ikisini tek pakette birleştirdiği için "backbone"un içinde görünür.
Head'ler yalnızca neck'in **çıktı sözleşmesini** tüketir: `{"0","1","2","3","pool"}` (strides
4/8/16/32/64, 256 kanal). `build_backbone(name, ...)` (`models/backbone.py`) şu seçenekleri verir:

| `backbone_name` | gövde | neck |
|---|---|---|
| `resnet50` | ResNet50 | FPN |
| `dino_vitb16` | DINOv1 ViT-B/16 | Simple Feature Pyramid (`models/dino_backbone.py`) |
| `dinov2_vitb14`(`_reg`) | DINOv2 ViT-B/14 (opsiyonel 4 register) | Simple Feature Pyramid (`models/dinov2_backbone.py`) |
| `dinov2_vits14`(`_reg`) | DINOv2 ViT-S/14 (opsiyonel 4 register) | Simple Feature Pyramid (`models/dinov2_backbone.py`) |

DINOv2 (patch14), DINOv1 baseline'ını bozmamak için ayrı dosyada (`dinov2_backbone.py`); neck
mantığı ortaktır, yalnızca patch/model farklıdır. DINOv2 boyutları (isim → timm modeli) o
dosyadaki `DINOV2_MODELS` registry'sinde; yeni boyut eklemek tek satır (embed_dim ve register
token'ları gövdeden dinamik okunur). patch14 girişleri 14'e bölünebilmeli (config'te 518).

DINO (self-supervised ViT) *plain* bir transformer, tek çözünürlük (stride 16) üretir; FPN'in
birleştireceği doğal hiyerarşi olmadığından FPN yerine ViTDet'in Simple Feature Pyramid'i tek
haritadan 5 seviyeyi türetir. İkisi de aynı sözleşmeyi ürettiği için üç head + RetinaNet
değişmeden çalışır. Gerekçeler: [EXPERIMENTS.md](EXPERIMENTS.md) "Kararlar" notu.

**Neden RetinaNet, neden Mask R-CNN değil:** `MaskRCNN`, RPN+RoIAlign+kutu/mask head'lerini kendi `forward`'ının içine gömüyor — 4. bir görevi (classification) oraya eklemek internal koda müdahale gerektirirdi. `RetinaNet` ise `backbone → feature dict → head` sınırını temiz bırakıyor, üç kafa da aynı dict'i okuyor. Bu yüzden segmentation şu an **semantic** (piksel başına tek sınıf haritası), COCO'daki gibi **instance** segmentation değil — bilinçli bir v1 kararı (`models/maskrcnn_v2.py`'de v2 yol haritası yazılı).

---

## 2. Klasör ağacı ve her parçanın rolü

```
MTL/
├── backbone.py              # kök dizindeki eski dosya; artık sadece yönlendirme (shim)
│                             #   from mtl.models.backbone import build_backbone
├── pyproject.toml            # src/ altını "mtl" adında pip paketine çevirir (pip install -e .)
├── requirements-local.txt    # local CPU ortamı için sabitlenmiş paket listesi
├── requirements-colab.txt    # Colab için notlar (torch zaten kurulu geliyor orada)
├── README.md                  # NASIL çalıştırılır
├── ARCHITECTURE.md            # BU DOSYA — sistem neden böyle
│
├── src/mtl/                   # ← asıl kütüphane, her şey buradan import ediliyor
│   ├── config.py               # tüm ayarların Python karşılığı (aşağıda detaylı)
│   ├── models/                 # "NE öğreniliyor": backbone + 3 head + birleştirici
│   │   ├── backbone.py           # build_backbone: gövde+neck kurar (resnet50→FPN | dino_vitb16→SFP)
│   │   ├── dino_backbone.py      # DINOv1 ViT-B/16 + Simple Feature Pyramid (ResNet+FPN ile aynı arayüz)
│   │   ├── detection_head.py     # RetinaNet kurar
│   │   ├── segmentation_head.py  # FCN tabanlı piksel-sınıflandırma head'i
│   │   ├── classification_head.py# GAP+FC çok-etiketli sınıflandırma head'i
│   │   ├── multitask_model.py    # ← KALP: backbone'u 1 kez çalıştırıp 3 head'e dağıtan sınıf
│   │   └── maskrcnn_v2.py         # kod değil, v2 (instance segmentation) yol haritası notu
│   ├── losses/
│   │   └── joint_loss.py          # 4 loss'u tek ağırlıklı toplama indirger
│   ├── datasets/                  # "COCO JSON" ile "PyTorch tensor" arasındaki köprü
│   │   ├── coco_subset.py           # tam COCO'dan ~22.5k'lık alt küme seçer
│   │   ├── coco_multitask.py        # ← her __getitem__ 3 görev için de etiket üretir
│   │   ├── transforms.py             # image+box+mask'i BİRLİKTE resize/flip eder
│   │   └── collate.py                 # DataLoader'ın batch'leri nasıl yığacağı
│   ├── engine/                     # "NASIL öğreniliyor": döngü, ölçüm, kayıt
│   │   ├── train_one_epoch.py        # forward→loss→backward→step döngüsü + periyodik checkpoint
│   │   ├── evaluate.py                # mAP (detection) / mIoU (seg) / mAP-F1 (classification)
│   │   └── checkpoint.py              # model+optimizer state'ini diske yazma/okuma
│   └── utils/
│       ├── device.py                  # "cuda" istenip yoksa CPU'ya düş
│       ├── seed.py                     # tekrarlanabilirlik
│       └── logging.py                  # adım başına loss'ları stdout+CSV'ye yaz
│
├── configs/                    # YAML dosyaları — KOD DEĞİL, sadece parametre setleri
│   ├── smoke_cpu.yaml            # local, 20 görüntü, 2 adım, CPU
│   └── train_colab_gpu.yaml       # gerçek koşu, 22.500 görüntü, 16 epoch, CUDA
│
├── scripts/                    # çalıştırılabilir komutlar (terminalden/Colab'dan çağrılır)
│   ├── prepare_coco_subset.py    # tam COCO → alt küme JSON
│   ├── download_subset_images.py # alt kümenin sadece ihtiyaç duyduğu görüntüleri indirir
│   ├── train.py                    # ← ana eğitim CLI'ı
│   └── eval.py                     # checkpoint'ten metrik hesaplar
│
├── notebooks/
│   └── colab_train.ipynb        # scripts/ altındakileri sırayla çağıran Colab defteri
│
├── tests/                       # kod doğruluğunu kanıtlayan otomatik testler
│   ├── test_smoke_forward.py      # sahte veriyle overfit-one-batch sanity check
│   └── test_dataset_shapes.py      # dataset çıktı shape/dtype doğrulaması
│
└── data/, checkpoints/, runs/    # gitignored — üretilen veri/ağırlık/log burada durur
```

**Kural:** `models/` "model ne hesaplar" sorusuna, `engine/` "modeli nasıl eğitir/ölçeriz" sorusuna cevap veriyor. `configs/` ise Python değil — sadece sayı/yol içeren metin dosyaları, `scripts/` onları okuyup `src/mtl/`'i çağırıyor.

---

## 3. Bir görüntünün baştan sona yolculuğu

```
1) COCO annotation JSON  ──►  CocoMultiTaskDataset.__getitem__   (datasets/coco_multitask.py)
   │  Tek görüntü için 3 farklı etiket üretir:
   │    boxes, labels     (detection)
   │    sem_mask           (segmentation, annToMask ile rasterize)
   │    cls_labels          (classification, çok-etiketli vektör)
   ▼
2) build_transforms()  ──►  transforms.py
   │  Görüntü + kutular + mask AYNI ANDA sabit boyuta resize edilir
   │  (biri değişince diğerleri de tutarlı kalmalı — ayrı ayrı transform edilemez)
   ▼
3) DataLoader + collate_fn  ──►  datasets/collate.py
   │  Birden fazla örnek tek batch tensor'üne yığılır: (B,3,H,W)
   ▼
4) MultiTaskModel.forward()  ──►  models/multitask_model.py
   │  a) self.backbone(images)        → features dict {"0","1","2","3"}  (TEK forward pass)
   │  b) detection_model.head(...)    → classification + bbox_regression loss
   │  c) seg_head(features["0"])      → seg_loss
   │  d) cls_head(features["3"])      → cls_loss
   ▼
5) combine_losses()  ──►  losses/joint_loss.py
   │  total = 1.0·classification + 1.0·bbox_regression + 1.0·seg_loss + 0.5·cls_loss
   ▼
6) total.backward() + optimizer.step()  ──►  engine/train_one_epoch.py
   │  Ağırlıklar güncellenir. Her log_every adımda CsvLogger'a yazılır.
   │  Her checkpoint_every_steps adımda (varsa) ara checkpoint kaydedilir.
   ▼
7) Epoch bitince  ──►  engine/checkpoint.py save_checkpoint()
      {run_name}_epoch{N}.pt dosyasına model+optimizer state'i yazılır.
```

Değerlendirme (eval) aynı 1-4 adımlarını `model.eval()` modunda çalıştırır (loss yerine tahmin döner: kutular, seg haritası, sınıf olasılıkları), sonra `engine/evaluate.py` bunları gerçek etiketlerle kıyaslayıp mAP/mIoU/F1 hesaplar.

---

## 4. Dosyalar arası bağımlılık (kim kimi import ediyor)

```
scripts/train.py
  ├─► mtl.config                 (load_config, apply_smoke_overrides)
  ├─► mtl.datasets.coco_multitask
  ├─► mtl.datasets.collate
  ├─► mtl.engine.checkpoint       (save_checkpoint, load_checkpoint)
  ├─► mtl.engine.train_one_epoch
  ├─► mtl.models.multitask_model
  └─► mtl.utils.{device,logging,seed}

mtl.models.multitask_model
  ├─► mtl.models.backbone          (build_backbone)
  ├─► mtl.models.detection_head    (build_detection_model)
  ├─► mtl.models.segmentation_head (SemanticSegHead)
  └─► mtl.models.classification_head (MultiLabelClsHead)

mtl.engine.train_one_epoch
  ├─► mtl.losses.joint_loss        (combine_losses)
  └─► mtl.engine.checkpoint         (periyodik kayıt için save_checkpoint)

mtl.datasets.coco_multitask
  └─► mtl.datasets.transforms      (build_transforms)
```

Yani `multitask_model.py` mimarinin **kalbi** — dört model dosyasını (backbone + 3 head) birbirine bağlayan tek yer. `train.py` ise dış dünyaya açılan **kapı** — config okuyup dataset/model/optimizer'ı kurup `train_one_epoch`'u çağıran orkestratör.

---

## 5. Config sistemi nasıl işliyor

`src/mtl/config.py` dört dataclass tanımlıyor: `DataConfig`, `ModelConfig`, `LossConfig`, `TrainConfig` (hepsi `Config` içinde toplanıyor). `load_config(path, overrides)`:

1. Önce dataclass'ların **varsayılan değerleriyle** bir `Config` oluşturur.
2. `--config configs/X.yaml` dosyasını okuyup üzerine yazar.
3. `--overrides section.field=value` argümanlarını (varsa) en son uygular — yani CLI, YAML'ı ezer.

`smoke_cpu.yaml` ile `train_colab_gpu.yaml` arasında **kod farkı yok**, sadece bu dosyalardaki sayı/yol farklı. `--smoke` bayrağı da `apply_smoke_overrides()` ile hangi config yüklenirse yüklensin `max_steps=2, n_images≤20` zorluyor — Colab config'ini bile local'de güvenle "kuru deneme" yapabilmek için.

---

## 6. Şu anki v1 sınırları (bilerek)

- **Segmentation semantic, instance değil** (yukarıda anlatıldı) — v2'de `MaskRCNN`'e geçiş `maskrcnn_v2.py`'de planlı.
- **Loss ağırlıkları sabit** (`det_cls=1, det_box=1, seg=1, cls=0.5`), config'te elle ayarlanıyor. Adaptif ağırlıklandırma (Kendall et al. 2018, uncertainty weighting) v2 notu olarak `joint_loss.py`'de yazılı.
- **Resume, tam checkpoint'ten değil ağırlık bazlı** — `--resume`, dataloader'ın kaldığı yeri değil, sadece model+optimizer ağırlıklarını geri yükler; eğitim epoch başından yeniden dataloader'ı gezer. Basit ve pratik, ama "tam olarak kaldığı adımdan devam" değil.
- **`iscrowd` annotasyonları tamamen atlanıyor** (hem detection hem segmentation'da) — COCO'daki "kalabalık bölge" etiketleri için doğru davranış (ignore-region) yerine v1'de basitçe yok sayılıyor.

Bu sınırların hepsi bilinçli, "önce çalışan bir sistem, sonra iyileştirme" mantığıyla alınmış kararlar — plan dosyasında gerekçeleriyle birlikte duruyor.
