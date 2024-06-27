# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import roi_align

try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_

from .transformer import PatchDropout
from .rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        from paddle.incubate.checkpoint import checkpoint
else:
    from paddle.incubate.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")
from typing import Sequence


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Layer):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        drop=0.,
        subln=False,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the original BERT implement 
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Silu, drop=0., 
                norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias_attr=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias_attr=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias_attr=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=False)

        if qkv_bias:
            self.q_bias = self.create_parameter(shape=[all_head_dim], default_initializer=nn.initializer.Constant(0.0))
            self.v_bias = self.create_parameter(shape=[all_head_dim], default_initializer=nn.initializer.Constant(0.0))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = self.create_parameter(
                shape=[self.num_relative_distance, num_heads], 
                default_initializer=nn.initializer.Constant(0.0))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(window_size[0])
            coords_w = paddle.arange(window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = paddle.zeros([window_size[0] * window_size[1] + 1] * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln: 
            q = F.linear(x, self.q_proj.weight, bias=self.q_bias)
            k = F.linear(x, self.k_proj.weight, bias=None)
            v = F.linear(x, self.v_proj.weight, bias=self.v_bias)

            q = q.reshape([B, N, self.num_heads, -1]).transpose([0, 2, 1, 3])  # B, num_heads, N, C
            k = k.reshape([B, N, self.num_heads, -1]).transpose([0, 2, 1, 3])
            v = v.reshape([B, N, self.num_heads, -1]).transpose([0, 2, 1, 3])
        else: 
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = paddle.concat([self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias])
            
            qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])  # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            if attn_mask is not None:
                attn_mask = attn_mask.to(q)
            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = paddle.concat([q[:, :, :1, :], ro_q_t], axis=-2).astype(v.dtype)
            k = self.rope(k).astype(v.dtype)
        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape(
                [self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        if attn_mask is not None:
            attn_mask = attn_mask.astype(attn.dtype)
            attn = attn.masked_fill(attn_mask, float('-inf'))
            attn = F.softmax(attn, axis=-1).masked_fill(attn_mask, 0.)
        else:
            attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        if self.xattn:
            x = xops.memory_efficient_attention(q, k, v, p=self.xattn_drop, scale=self.scale)
        else:
            x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.inner_attn_ln(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop=0., 
        attn_drop=0.,
        drop_path=0., 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        window_size=None, 
        attn_head_dim=None,
        xattn=False, 
        rope=None, 
        subln=False,
        ffn=True,
        ffn_ln=False,
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop, 
            window_size=window_size, 
            attn_head_dim=attn_head_dim,
            xattn=xattn,
            rope=rope,
            subln=subln,
            norm_layer=norm_layer,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        if ffn:
            if ffn_ln:
                self.mlp = SwiGLU(
                    in_features=dim, 
                    hidden_features=int(dim * mlp_ratio), 
                    act_layer=act_layer, 
                    drop=drop,
                    norm_layer=norm_layer, 
                    subln=subln,
                    )
            else:
                self.mlp = Mlp(
                    in_features=dim, 
                    hidden_features=int(dim * mlp_ratio), 
                    act_layer=act_layer, 
                    drop=drop,
                    norm_layer=norm_layer, 
                    subln=subln,
                    )
        else:
            self.mlp = nn.Identity()

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Layer):
    def __init__(self, window_size: Sequence[int], num_heads: int):
        super().__init__()
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            shape=[self.num_relative_distance, num_heads], 
            default_initializer=nn.initializer.Constant(0.0))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = paddle.zeros([self.window_size[0] * self.window_size[1] + 1] * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape(
            [self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias


class PatchEmbed(nn.Layer):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 xattn=False, rope=None, subln=False, ffn=True, ffn_ln=False,
                 use_mean_pooling=True, init_values=None,
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_mean_pooling = use_mean_pooling
        self.norm_layer = norm_layer if norm_layer else partial(nn.LayerNorm, epsilon=1e-6)
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(shape=[1, 1, embed_dim], default_initializer=nn.initializer.Constant(0))
        self.pos_embed = self.create_parameter(shape=[1, num_patches + 1, embed_dim], default_initializer=nn.initializer.Constant(0))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer,
                xattn=xattn, rope=rope, subln=subln, ffn=ffn, ffn_ln=ffn_ln,
                window_size=None)
            for i in range(depth)])
        self.norm = self.norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand([B, -1, -1])  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)
        x = self.pos_drop(x + self.pos_embed)
        rel_pos_bias = None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        if self.use_mean_pooling:
            x = x[:, 1:].mean(axis=1)  # global pool without cls token
        else:
            x = x[:, 0]

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class VisionTransformerRelativePositionBias(VisionTransformer):
    def __init__(self, window_size=None, **kwargs):
        super().__init__(**kwargs)
        assert window_size is not None
        self.rel_pos_bias = RelativePositionBias(window_size=window_size, num_heads=kwargs['num_heads'])

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand([B, -1, -1])  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)
        x = self.pos_drop(x + self.pos_embed)

        rel_pos_bias = self.rel_pos_bias()
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        if self.use_mean_pooling:
            x = x[:, 1:].mean(axis=1)  # global pool without cls token
        else:
            x = x[:, 0]

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
