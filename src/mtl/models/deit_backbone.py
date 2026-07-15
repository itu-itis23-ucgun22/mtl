"""DeiT (Data-efficient image Transformer, Touvron et al. 2021) gövdesi + Simple Feature
Pyramid neck'i. Foundation-model sweep'in "supervised ViT" temsilcisi (ROADMAP Faz 1).

DeiT = ImageNet-1k üzerinde SUPERVISED (sınıflandırma etiketiyle) eğitilmiş bir ViT-B/16.
Sweep'te "supervised" paradigmasını şu ana kadar ResNet temsil ediyordu — ama ResNet conv+FPN,
diğerleri ViT+SFP → "supervised vs SSL" kıyası conv/FPN confound'uyla karışıktı. DeiT bunu kapatır:
ViT-B/16 @512 -> 32x32 grid ile DINOv1/MAE/CLIP/SAM ile HER EKSENDE aynı (boyut, patch, grid, norm,
neck, head, batch). Böylece "supervised" ile "SSL/dil/seg-native" AYNI mimaride kıyaslanabilir hale gelir.

⭐ Adil çekirdeğin supervised ayağı — DINOv1 vs DeiT = "aynı ViT, SSL vs supervised pretraining".

Not: `deit_base_patch16_224.fb_in1k` = SAF supervised DeiT (distillation'lı `deit_*_distilled` DEĞİL;
distilled varyant öğretmen-öğrenci kullanır, saf supervised paradigmayı bulandırırdı). Normalizasyon
ImageNet (CLIP'teki gibi renorm GEREKMEZ). Yerel ağırlık yükleme: bkz. models/timm_weights.py.
"""
from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

from mtl.models.sfp import SimpleFeaturePyramid
from mtl.models.timm_weights import create_timm_model

DEIT_MODEL = "deit_base_patch16_224.fb_in1k"  # supervised DeiT ViT-B/16 (in1k); distilled DEĞİL
PATCH_SIZE = 16
OUT_CHANNELS = 256


class DeitBackbone(nn.Module):
    """DeiT ViT-B/16 (supervised) + Simple Feature Pyramid, BackboneWithFPN ile aynı arayüz.

    forward(images) -> OrderedDict {"0","1","2","3","pool"} (strides 4/8/16/32/64, 256 kanal).
    trainable_blocks: 0 = ViT tamamen donuk (kanonik sweep), N = son N blok + final norm.
    """

    def __init__(
        self,
        pretrained: bool = True,
        trainable_blocks: int = 0,
        out_channels: int = OUT_CHANNELS,
    ):
        super().__init__()
        self.vit = create_timm_model(
            DEIT_MODEL,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,  # 224 disi girislerde pos-embed interpolasyonu (512 -> 32x32)
        )
        self.num_prefix_tokens = self.vit.num_prefix_tokens  # CLS token(lar)i atmak icin
        self.out_channels = out_channels
        self._set_trainable_blocks(trainable_blocks)
        self.sfp = SimpleFeaturePyramid(self.vit.embed_dim, out_channels)  # embed_dim = 768

    def _set_trainable_blocks(self, trainable_blocks: int) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        if trainable_blocks and trainable_blocks > 0:
            blocks = self.vit.blocks
            for blk in blocks[-min(trainable_blocks, len(blocks)):]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.vit.norm.parameters():
                p.requires_grad = True

    def _tokens_to_grid(self, images: Tensor) -> Tensor:
        """ViT patch token'larini (B, embed, h, w) uzamsal haritaya cevirir."""
        b, _, h_img, w_img = images.shape
        tokens = self.vit.forward_features(images)             # (B, prefix+N, embed)
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]   # CLS'i at -> (B, N, embed)
        h, w = h_img // PATCH_SIZE, w_img // PATCH_SIZE
        return patch_tokens.transpose(1, 2).reshape(b, -1, h, w)

    def trunk_forward(self, images: Tensor) -> Tensor:
        """DONUK ViT govdesinin ciktisi: (B, 768, 32, 32). Feature-caching icin (donukken deterministik)."""
        return self._tokens_to_grid(images)

    def neck_forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.sfp(x)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.sfp(self._tokens_to_grid(images))
