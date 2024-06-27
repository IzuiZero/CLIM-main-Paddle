import os
import logging
import math
from collections import OrderedDict
from typing import Callable, Optional, Sequence
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except:
    from timm.layers import trunc_normal_

from .rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast
from .utils import to_2tuple

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        import deepspeed
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        print("Please 'pip install deepspeed'")
        deepspeed = None
        from paddle.fluid.dygraph.checkpoint import checkpoint
else:
    from paddle.fluid.dygraph.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")

class LayerNormFp32(nn.LayerNorm):
    """Subclass PaddlePaddle's LayerNorm to handle fp16 (by casting to float32 and back)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: paddle.Tensor):
        output = F.layer_norm(
            x.astype('float32'),
            self.normalized_shape,
            self.weight.astype('float32') if self.weight is not None else None,
            self.bias.astype('float32') if self.bias is not None else None,
            self.eps,
        )
        return output.astype(x.dtype)


class LayerNorm(nn.LayerNorm):
    """Subclass PaddlePaddle's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.astype(orig_type)

class QuickGELU(nn.Layer):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: paddle.Tensor):
        return x * F.sigmoid(1.702 * x)


class LayerScale(nn.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = paddle.create_parameter(shape=[dim], dtype='float32', default_initializer=paddle.nn.initializer.Constant(init_values))

    def forward(self, x):
        return x * self.gamma if self.inplace else x * self.gamma

class PatchDropout(nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = paddle.jit.to_static(x[:, :1])

        batch = x.shape[0]
        num_tokens = x.shape[1]

        batch_indices = paddle.arange(batch)
        batch_indices = batch_indices.unsqueeze(-1)

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = paddle.randn([batch, num_tokens])
        patch_indices_keep = paddle.topk(rand, k=num_patches_keep, axis=-1, descending=False).indices

        x = paddle.index_select(x, batch_indices, patch_indices_keep)

        if self.exclude_first_token:
            x = paddle.concat((cls_tokens, x), axis=1)

        if self.training and os.getenv('RoPE') == '1':
            return x, patch_indices_keep

        return x


def _in_projection_packed(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    w: paddle.Tensor,
    b: Optional[paddle.Tensor] = None,
    ):
    """
    https://github.com/pytorch/pytorch/blob/db2a237763eb8693a20788be94f8c192e762baa8/torch/nn/functional.py#L4726
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, axis=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = paddle.split(w, [E, E * 2], axis=-1)
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = paddle.split(b, [E, E * 2], axis=-1)
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, axis=-1)
    else:
        w_q, w_k, w_v = paddle.split(w, 3, axis=-1)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = paddle.split(b, 3, axis=-1)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

class Attention(nn.Layer):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            xattn=False,
            rope=False
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = self.create_parameter(shape=[dim * 3, dim], default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=self.scale))
        if qkv_bias:
            self.in_proj_bias = self.create_parameter(shape=[dim * 3], default_initializer=paddle.nn.initializer.Constant(0.0))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = self.create_parameter(shape=[num_heads, 1, 1], default_initializer=paddle.nn.initializer.Constant(math.log(10)))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = self.create_parameter(shape=[num_heads, 1, 1], default_initializer=paddle.nn.initializer.Constant(1.0))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop
        self.rope = rope

    def forward(self, x, attn_mask: Optional[paddle.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, axis=-1)
        if self.xattn:
            q = q.transpose([1, 0, 2]).reshape([N, L, self.num_heads, -1])
            k = k.transpose([1, 0, 2]).reshape([N, L, self.num_heads, -1])
            v = v.transpose([1, 0, 2]).reshape([N, L, self.num_heads, -1])

            x = xops.memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop,
                scale=self.scale if self.logit_scale is None else None,
                attn_bias=xops.LowerTriangularMask() if attn_mask is not None else None,
                )
        else:
            q = q.transpose([1, 0, 2]).reshape([N * self.num_heads, L, -1])
            k = k.transpose([1, 0, 2]).reshape([N * self.num_heads, L, -1])
            v = v.transpose([1, 0, 2]).reshape([N * self.num_heads, L, -1])

            if self.logit_scale is not None:
                attn = paddle.matmul(F.norm(q, axis=-1), F.norm(k, axis=-1).transpose([0, 2, 1]))
                logit_scale = F.clip(self.logit_scale, max=self.logit_scale_max).exp()
                attn = (attn * logit_scale).softmax(-1)
                attn = attn.masked_fill_(attn_mask, 0.)
                attn = self.attn_drop(attn)
                out = paddle.matmul(attn, v)
            else:
                out = F.matmul(q, k.transpose([0, 2, 1]))
                out = out.masked_fill_(attn_mask, float('-inf')).softmax(-1)
                out = self.attn_drop(out)
                out = F.matmul(out, v)

            out = out.reshape([N, self.num_heads, L, -1])
            out = out.transpose([1, 0, 2, 3])
            out = out.reshape([N, L, -1])
        return self.out_drop(self.out_proj(out))

def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)

class rPE3D(nn.Layer):
    def __init__(self, max_len, embed_dim, num_heads, drop_path, use_rotary_emb, rotary_kwargs):
        super().__init__()
        self.pe = PositionalEncoding3D(
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path=drop_path,
            use_rotary_emb=use_rotary_emb,
            rotary_kwargs=rotary_kwargs,
        )

    def forward(self, x):
        return self.pe(x)


class Block(nn.Layer):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            use_checkpoint=False,
            use_rotary_emb=False,
            rotary_kwargs=dict(),
            )

        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            scale_heads=qk_scale,
            **rotary_kwargs,
        )

        self.drop_path = drop_path

        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=QuickGELU, drop=drop)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = x[0] + self.drop_path

        x = self.norm2(x)
        x = self.mlp(x)

        return x


class rPEEmbedding(nn.Layer):
    def __init__(self, args, conv2d = None, div, dim, mask = True):
        super().__init__()
        self.args = args
        self.proj = torch.nn
        self.embed_proj = None
        self.args = div
        self.args = dim
        self.seq_length = args
        self.mask = mask

    def forward(self, x):
        return x

class rPEAttention(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., use_checkpoint=False, use_rotary_emb=False, rotary_kwargs=dict(),):

        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            scale_heads=qk_scale,
            **rotary_kwargs,
        )

        self.drop_path = drop_path

        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=QuickGELU, drop=drop)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = x[0] + self.drop_path

        x = self.norm2(x)
        x = self.mlp(x)

        return x


class Posix nn.LayerNorm:
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., use_checkpoint=False, use_rotary_emb=False, rotary_kwargs=dict(),):

        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            scale_heads=qk_scale,
            **rotary_kwargs,
        )

        self.drop_path = drop_path

        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=QuickGELU, drop=drop)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = x[0] + self.drop_path

        x = self.norm2(x)
        x = self.mlp(x)

        return x


class LayerScale1d(nn.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = paddle.create_parameter(shape=[dim], dtype='float32', default_initializer=paddle.nn.initializer.Constant(init_values))

    def forward(self, x):
        return x * self.gamma if self.inplace else x * self.gamma


class PatchDropout1d(nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = paddle.jit.to_static(x[:, :1])

        batch = x.shape[0]
        num_tokens = x.shape[1]

        batch_indices = paddle.arange(batch)
        batch_indices = batch_indices.unsqueeze(-1)

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = paddle.randn([batch, num_tokens])
        patch_indices_keep = paddle.topk(rand, k=num_patches_keep, axis=-1, descending=False).indices

        x = paddle.index_select(x, batch_indices, patch_indices_keep)

        if self.exclude_first_token:
            x = paddle.concat((cls_tokens, x), axis=1)

        if self.training and os.getenv('RoPE') == '1':
            return x, patch_indices_keep

        return x


def _in_projection_packed(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    w: paddle.Tensor,
    b: Optional[paddle.Tensor] = None,
    ):
    """
    https://github.com/pytorch/pytorch/blob/db2a237763eb8693a20788be94f8c192e762baa8/torch/nn/functional.py#L4726
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, axis=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = paddle.split(w, [E, E * 2], axis=-1)
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = paddle.split(b, [E, E * 2], axis=-1)
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, axis=-1)
    else:
        w_q, w_k, w_v = paddle.split(w, 3, axis=-1)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = paddle.split(b, 3, axis=-1)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

class Attention(nn.Layer):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            xattn=False,
            rope=False
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = self.create_parameter(shape=[3 * dim, dim], dtype="float32")
        self.in_proj_bias = self.create_parameter(shape=[3 * dim], dtype="float32")
        self.drop = nn.Dropout(attn_drop)

        if self.scaled_cosine:
            self.rescale = nn.Parameter(self.logit_scale_max * paddle.randn(1), requires_grad=True)
            self.logit_scale = nn.Parameter((1.0 / math.log(2.0)) * paddle.ones(1), requires_grad=True)

    def forward(
            self,
            q: paddle.Tensor,
            k: paddle.Tensor,
            v: paddle.Tensor,
            attn_mask: Optional[paddle.Tensor] = None,
            key_padding_mask: Optional[paddle.Tensor] = None,
            need_weights=True,
            )

        if not self.scaled_cosine:
            return multi_head_attention_forward(
                query=q,
                key=k,
                value=v,
                in_proj_weight=self.in_proj_weight,
                in_proj_bias=self.in_proj_bias,
                bias_k=None,
                bias_v=None,
                dropout_p=self.drop,
                out_proj_weight=self.out_proj_weight,
                out_proj_bias=self.out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                )

        else:
            return multi_head_attention_forward(
                query=q,
                key=k,
                value=v,
                rescale=self.rescale,
                logit_scale=self.logit_scale,
                dropout_p=self.drop,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )

class Encoder(nn.Layer):
    def __init__(self, max_len, embed_dim, num_heads, mlp_ratio, drop_rate, num_blocks, use_rotary_emb=False, rotary_kwargs=None):
        super().__init__()
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.LayerList([Block(embed_dim, num_heads, mlp_ratio, drop_rate, use_rotary_emb, rotary_kwargs) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.encoder_norm(x)

class xhattn(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        use_checkpoint=False,
        vvv,
        in_features,
        hidden_features,
        act_layer,
        pool,):
        super().__init__()
        self.pool = pool
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = xattn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            use_checkpoint=use_checkpoint,
            vvv=vvv,
        )
        self.drop_path = drop_path
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.vvv = vvv

    def forward(self, x):
        x = self.norm1(x)
        x = self.qkv(x)
        x = x[0] + self.drop_path
        x = self.norm2(x)
        x = self.mlp(x)
        return x

class AttentivePooling(nn.Layer):
    def __init__(self, dim, dim_out, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., drop_path=0., act_layer=quick_gelu):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, scale_heads=False)
        self.proj = Linear(dim, dim_out)

    def forward(self, x):
        x = self.norm(x)
        x = self.attn(x)
        x = self.proj(x)
        return x

class Attn1Dx1D(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., drop_path=0., act_layer=quick_gelu):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, scale_heads=False)
        self.proj = Linear(dim, dim_out)

    def forward(self, x):
        x = self.norm(x)
        x = self.attn(x)
        x = self.proj(x)
        return x
