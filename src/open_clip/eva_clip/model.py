""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

try:
    from .hf_model import HFTextEncoder
except:
    HFTextEncoder = None
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .eva_vit_model import EVAVisionTransformer
from .transformer import LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer

try:
    from apex.normalization import FusedLayerNorm
except:
    FusedLayerNorm = LayerNorm
    print("Please 'pip install apex'")

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0. # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    drop_path_rate: Optional[float] = None  # drop path rate
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    eva_model_name: str = None # a valid eva model name overrides layers, width, patch_size
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16   # 224/14
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    masked_language_modeling: bool = False
    fusedLN: bool = False
    xattn: bool = False
    attn_mask: bool = True

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = paddle.bfloat16
    elif precision == 'fp16':
        cast_dtype = paddle.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[paddle.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNorm
        
        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool, #False
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer= partial(FusedLayerNorm, epsilon=1e-6) if vision_cfg.fusedLN else partial(norm_layer, epsilon=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len= vision_cfg.pt_hw_seq_len,   # 224/14
            intp_freq= vision_cfg.intp_freq,
            naiveswiglu= vision_cfg.naiveswiglu,
            subln= vision_cfg.subln
        )
    elif vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size
        )
        act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (paddle.float16, paddle.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            global_average_pool=vision_cfg.global_average_pool,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[paddle.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            tokenizer_name=text_cfg.hf_tokenizer_name,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            masked_language_modeling=text_cfg.masked_language_modeling
       )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer= FusedLayerNorm if text_cfg.fusedLN else norm_layer,
            xattn=text_cfg.xattn,
            attn_mask=text_cfg.attn_mask,
        )
    return text


class CLIP(nn.Layer):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[paddle.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.embed_dim = embed_dim
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.logit_scale = text.logit_scale
        self.cast_dtype = cast_dtype

        if self.cast_dtype:
            self.token_embedding = self.token_embedding.cast(self.cast_dtype)
            self.positional_embedding = self.positional_embedding.cast(self.cast_dtype)
            self.ln_final = self.ln_final.cast(self.cast_dtype)
            self.text_projection = self.text_projection.cast(self.cast_dtype)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.astype(self.visual.conv1.weight.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).astype(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.astype(self.dtype)
        x = x.transpose([1, 0, 2])  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose([1, 0, 2])  # LND -> NLD

        x = self.ln_final(x).astype(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        x = x[paddle.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / text_features.norm(axis=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def get_cast_dtype(self):
        return self.cast_dtype


def convert_weights_to_lp(model: nn.Layer):
    """
    Convert applicable model parameters to lower precision (bf16 or fp16)
    """
    def _convert_weights_to_lp(l):
        if isinstance(l, (nn.Linear, nn.Conv2D)) and l.weight.dtype not in (paddle.float32, paddle.float64):
            l.weight.set_value(l.weight.cast(l.weight.dtype))
            if l.bias is not None:
                l.bias.set_value(l.bias.cast(l.weight.dtype))
    model.apply(_convert_weights_to_lp)


def build_model(state_dict: dict, cast_dtype: Optional[paddle.dtype] = None):
    """ 
    Build a CLIP model from the original model state dict
    """
    vision_cfg_keys = [k for k in state_dict.keys() if k.startswith('visual.') and 'visual.proj' not in k]
    text_cfg_keys = [k for k in state_dict.keys() if k.startswith('transformer.') or 'positional_embedding' in k]
    cast_dtype = get_cast_dtype(cast_dtype)

    vision_cfg = {k.replace('visual.', ''): state_dict.pop(k) for k in vision_cfg_keys}
    text_cfg = {k.replace('transformer.', ''): state_dict.pop(k) for k in text_cfg_keys}

    model = CLIP(
        embed_dim=state_dict['text_projection'].shape[1],
        vision_cfg=CLIPVisionCfg(**vision_cfg),
        text_cfg=CLIPTextCfg(**text_cfg),
        quick_gelu=False,
        cast_dtype=cast_dtype
    )
    convert_weights_to_lp(model)

    model.load_dict(state_dict)
    return model

