import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import roi_align
from collections import OrderedDict
import math

# Import utils if necessary, assuming it's not provided here

class LayerNormFp32(nn.LayerNorm):
    """Subclass paddle's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x):
        orig_type = x.dtype
        x = F.layer_norm(x.astype('float32'), self.normalized_shape, self.weight, self.bias, self.epsilon)
        return x.astype(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass paddle's LayerNorm (with cast back to input dtype)."""

    def forward(self, x):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.epsilon)
        return x.astype(orig_type)


class QuickGELU(nn.Layer):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x):
        return x * paddle.sigmoid(1.702 * x)


class LayerScale(nn.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = self.create_parameter(shape=[dim], default_initializer=paddle.nn.initializer.Constant(init_values))

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

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = paddle.jit.annotate(paddle.Tensor, x[:, :1])

        batch = x.shape[0]
        num_tokens = x.shape[1]

        batch_indices = paddle.arange(batch).unsqueeze(-1)

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = paddle.randn([batch, num_tokens])
        _, patch_indices_keep = paddle.topk(rand, num_patches_keep, axis=-1)

        x = paddle.gather(x, patch_indices_keep, axis=1)

        if self.exclude_first_token:
            x = paddle.concat((cls_tokens, x), axis=1)

        return x


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
            proj_drop=0.
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

    def forward(self, x, attn_mask=None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, axis=-1)
        q = q.transpose([1, 0, 2]).reshape([N * self.num_heads, L, -1])
        k = k.transpose([1, 0, 2]).reshape([N * self.num_heads, L, -1])
        v = v.transpose([1, 0, 2]).reshape([N * self.num_heads, L, -1])

        if self.logit_scale is not None:
            attn = paddle.matmul(F.layer_norm(q, normalized_shape=q.shape[1:], weight=None, bias=None, epsilon=1e-05), F.layer_norm(k, normalized_shape=k.shape[1:], weight=None, bias=None, epsilon=1e-05).transpose([0, 2, 1]))
            logit_scale = self.logit_scale.clamp(max=self.logit_scale_max).exp()
            attn = attn.reshape([N, self.num_heads, L, L]) * logit_scale
            attn = attn.reshape([-1, L, L])
        else:
            q = q * self.scale
            attn = paddle.matmul(q, k.transpose([0, 2, 1]))

        if attn_mask is not None:
            if attn_mask.dtype == paddle.bool:
                new_attn_mask = paddle.full_like(attn_mask, 0.0)
                attn_mask = paddle.where(attn_mask, new_attn_mask, attn_mask)
            attn += attn_mask

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v)
        if self.head_scale is not None:
            x = x.reshape([N, self.num_heads, L, C]) * self.head_scale
            x = x.reshape([-1, L, C])
        x = x.transpose([1, 0, 2]).reshape([L, N, C])
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Layer):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = self.create_parameter(shape=[n_queries, d_model], default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0))
        self.attn = nn.MultiHeadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: paddle.Tensor):
        x = self.ln_k(x.transpose([1, 0, 2]))  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.transpose([1, 0, 2])  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat([1, N, 1])


class ResidualAttentionBlock(nn.Layer):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: paddle.Tensor,
            k_x: Optional[paddle.Tensor] = None,
            v_x: Optional[paddle.Tensor] = None,
            attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        # attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: paddle.Tensor,
            k_x: Optional[paddle.Tensor] = None,
            v_x: Optional[paddle.Tensor] = None,
            attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class ResidualAttentionBlockV2(ResidualAttentionBlock):
    def proj_without_attn(self, value):
        attn_module = self.attn
        value = F.linear(value, attn_module.in_proj_weight,
                         bias=attn_module.in_proj_bias)[..., -attn_module.embed_dim:]
        value = F.linear(value, attn_module.out_proj.weight,
                         bias=attn_module.out_proj.bias)

        return value

    def forward_without_attn(self, q_x):
        x = q_x + self.ls_1(self.proj_without_attn(value=self.ln_1(q_x)))    # use the maskclip-zhou style
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Layer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.LayerList([
            ResidualAttentionBlockV2(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> paddle.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not paddle.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = paddle.checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def extract_feature_map(self, x, return_forward=False):
        for i in range(self.layers - 1):
            x = self.resblocks[i](x)
        x_forward = self.resblocks[-1](x)
        x = self.resblocks[-1].forward_without_attn(x)

        if return_forward:
            return x, x_forward
        else:
            return x

    def forward_image_dense(self, x, attn_mask):
        for i in range(self.layers - 1):
            x = self.resblocks[i](x, attn_mask=attn_mask)

        dense = self.resblocks[-1].forward_without_attn(x)
        image = self.resblocks[-1](x, attn_mask=attn_mask)

        return image, dense



class VisionTransformer(nn.Layer):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            n_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            input_patchnorm: bool = False,
            act_layer: callable = nn.GELU,
            norm_layer: callable = LayerNorm,
            output_tokens: bool = False
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = self.to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = self.to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm
        assert not input_patchnorm
        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2D(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias_attr=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = self.create_parameter((width,), default_initializer=Normal(mean=0.0, std=scale))
        self.positional_embedding = self.create_parameter((self.grid_size[0] * self.grid_size[1] + 1, width),
                                                          default_initializer=Normal(mean=0.0, std=scale))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = paddle.nn.Dropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = norm_layer(width)
        self.transformer = PaddleTransformer(
            dim=width,
            depth=layers,
            heads=heads,
            mlp_dim=int(width * mlp_ratio),
            norm_layer=norm_layer,
            act=act_layer
        )
        self.num_heads = heads

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = self.create_parameter((output_dim, output_dim), default_initializer=Normal(mean=0.0, std=scale))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = self.create_parameter((width, output_dim), default_initializer=Normal(mean=0.0, std=scale))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.trainable = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.ln_pre,
                ],
                self.positional_embedding,
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    # self.ln_post,     # fix layer norm
                ],
                # self.proj,        # fix output layers
            ]

            def _unlock(x):
                if isinstance(x, list):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, paddle.ParamAttr):
                        x.trainable = True
                    else:
                        for p in x.parameters():
                            p.trainable = True

            _unlock(groups[-unlocked_groups:])

    def attention_lock(self, **kwargs):
        for name, params in self.named_parameters():
            params.trainable = True if "attn" in name or "position" in name else False

    def init_parameters(self):
        pass

    @staticmethod
    def to_2tuple(x):
        return tuple([x, x]) if isinstance(x, int) else tuple(x)

    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: paddle.Tensor) -> tuple:
        if self.global_average_pool:
            return paddle.mean(x, axis=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: paddle.Tensor):

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.transpose((0, 2, 1))  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        cls_embed = paddle.unsqueeze(self.class_embedding.astype(x.dtype) + paddle.zeros(
            (x.shape[0], 1, x.shape[-1]), dtype=x.dtype, place=x.place), axis=1)  # shape = [*, 1, width]
        x = paddle.concat([cls_embed, x], axis=1)  # shape = [*, grid ** 2 + 1, width]

        if (h, w) == self.grid_size:
            pe = self.positional_embedding.astype(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.transpose((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose((1, 0, 2))  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = paddle.matmul(pooled, self.proj)

        if self.output_tokens:
            return pooled, tokens

        return pooled

    def post_attention(self, x):
        x = x.transpose((1, 0, 2))  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = paddle.matmul(pooled, self.proj)

        if self.output_tokens:
            return pooled, tokens

        return pooled

    def extract_roi_features(self, x, normed_boxes, extract_type='v2'):
        if extract_type == 'v1':
            return self._extract_roi_features_v1(x, normed_boxes)
        elif extract_type == 'v2':
            return self._extract_roi_features_v2(x, normed_boxes)
        else:
            raise NotImplementedError
            # assert extract_type == 'v3'
            # return self._extract_roi_features_v3(x, normed_boxes)

    def mask_pool(self, x, masks):
        feature_map = self.encode_dense(x)
        feature_map = F.normalize(feature_map, axis=-1)

        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = paddle.concat(masks, axis=0).astype('float32').reshape((len(masks), -1))    # bs, h*w
        feature_map = paddle.repeat(feature_map, repeats=paddle.to_tensor(num_masks_per_image, dtype='int64'),
                                    axis=0)
        features = paddle.sum(feature_map * paddle.unsqueeze(masks, axis=-1), axis=1) / (paddle.sum(masks, axis=1, keepdim=True) + 1e-12)

        return features

    def mask_features(self, x, masks):
        feature_map = self.encode_dense(x)
        feature_map = F.normalize(feature_map, axis=-1)

        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = paddle.concat(masks, axis=0).astype('float32').reshape((len(masks), -1))    # bs, h*w
        feature_map = paddle.repeat(feature_map, repeats=paddle.to_tensor(num_masks_per_image, dtype='int64'),
                                    axis=0)
        features = feature_map * paddle.unsqueeze(masks, axis=-1)

        return features

    def _extract_roi_features_v2(self, x, normed_boxes):
        # make boxes are normalized
        if not isinstance(normed_boxes, paddle.Tensor):
            normed_boxes = paddle.to_tensor(normed_boxes)
        x = self.encode_dense(x)
        # x: (batch_size, num_patches, embed_dim)
        h, w = self.image_size
        bboxes_scaled = normed_boxes * paddle.to_tensor([w, h, w, h], dtype='float32')
        # bboxes_scaled = normed_boxes
        box_features = paddle.empty([0, self.output_dim], dtype='float32')
        # box_features = box_features[0].cpu().numpy().item()
        for i, bbox in enumerate(bboxes_scaled):
            bx1, by1, bx2, by2 = bbox.cpu().numpy().tolist()
            if not (0 <= bx1 < bx2 <= w and 0 <= by1 < by2 <= h):
                continue
            int_boxes = paddle.to_tensor([bx1, by1, bx2, by2])
            int_boxes = paddle.maximum(int_boxes, paddle.to_tensor(0))
            int_boxes = paddle.minimum(int_boxes, paddle.to_tensor([w, h, w, h]))
            int_boxes = int_boxes.astype(paddle.long)
            box_patches = paddle.flatten(int_boxes)
            patch_pnorm = self.transformer(box_patches)
            box_features = paddle.cat([box_features, patch_pnorm], axis=0)
        return box_features

    def encode_dense(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.transpose((0, 2, 1))  # shape = [*, grid ** 2, width]
        cls_embed = paddle.unsqueeze(self.class_embedding.astype(x.dtype) + paddle.zeros(
            (x.shape[0], 1, x.shape[-1]), dtype=x.dtype, place=x.place), axis=1)  # shape = [*, 1, width]
        x = paddle.concat([cls_embed, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        if (h, w) == self.grid_size:
            pe = self.positional_embedding.astype(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)
        x = x + pe
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.transpose((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose((1, 0, 2))  # LND -> NLD
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        if self.proj is not None:
            pooled = paddle.matmul(pooled, self.proj)
        return pooled

    def encode_image(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.transpose((0, 2, 1))  # shape = [*, grid ** 2, width]
        cls_embed = paddle.unsqueeze(self.class_embedding.astype(x.dtype) + paddle.zeros(
            (x.shape[0], 1, x.shape[-1]), dtype=x.dtype, place=x.place), axis=1)  # shape = [*, 1, width]
        x = paddle.concat([cls_embed, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        if (h, w) == self.grid_size:
            pe = self.positional_embedding.astype(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)
        x = x + pe
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.transpose((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose((1, 0, 2))  # LND -> NLD
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        if self.proj is not None:
            pooled = paddle.matmul(pooled, self.proj)
        return pooled

    def rescale_positional_embedding(self, out_size, dtype):
        scale = out_size[0] * out_size[1] / (self.grid_size[0] * self.grid_size[1])
        target_size = paddle.cast(out_size, dtype)
        return F.interpolate(self.positional_embedding.astype(dtype).unsqueeze(0), target_size).reshape(
            (target_size[0], target_size[1], self.width))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(grid_size={self.grid_size}, image_size={self.image_size}, '
            f'output_dim={self.output_dim})'
        )


class DualPatchTransformer:
    def __init__(self, *args, **kwargs):
        self.encoder = VisionTransformer(*args, **kwargs)
        self.decoder = VisionTransformer(*args, **kwargs)

    def forward(self, src: paddle.Tensor, src_mask: paddle.Tensor, tgt: paddle.Tensor, tgt_mask: paddle.Tensor):
        src = self.encoder(src)
        # src shape = [*, dim]
        tgt = self.encoder(tgt)
        return src, tgt


def dual_patch_transformer(size, patch, layers, heads, mlpratio):
    return nn.Sequential(
        VisionTransformer(
            image_size=size,
            patch_size=patch,
            width=256,
            layers=layers,
            heads=heads,
            mlp_ratio=mlpratio,
            output_dim=256,
        ),
        nn.Flatten(1),
        nn.Linear(256 * (size // patch) ** 2, 1),
    )

def dual_patch_detector(num_classes, size, patch, layers, heads, mlpratio):
    return nn.Sequential(
        VisionTransformer(
            image_size=size,
            patch_size=patch,
            width=256,
            layers=layers,
            heads=heads,
            mlp_ratio=mlpratio,
            output_dim=256,
        ),
        nn.Conv2D(256, 2, kernel_size=3, stride=1, padding=1),
        nn.Conv2D(2, num_classes, kernel_size=1, stride=1),
        nn.Flatten(),
    )


class TextTransformer(nn.Layer):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: callable = nn.GELU,
        norm_layer: callable = LayerNorm,
        embed_cls: bool = False,
        pad_id: int = 0,
        output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = self.create_parameter((width, output_dim), default_initializer=Normal())
        
        if embed_cls:
            self.cls_emb = self.create_parameter((width,), default_initializer=Normal())
            self.num_pos = context_length + 1
        else:
            self.cls_emb = None
            self.num_pos = context_length

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = self.create_parameter((self.num_pos, width), default_initializer=Normal())
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        self.token_embedding.weight.set_value(paddle.randn(self.token_embedding.weight.shape) * 0.02)
        self.positional_embedding.set_value(paddle.randn(self.positional_embedding.shape) * 0.01)
        if self.cls_emb is not None:
            self.cls_emb.set_value(paddle.randn(self.cls_emb.shape) * 0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            block.attn.in_proj_weight.set_value(paddle.randn(block.attn.in_proj_weight.shape) * attn_std)
            block.attn.out_proj.weight.set_value(paddle.randn(block.attn.out_proj.weight.shape) * proj_std)
            block.mlp.c_fc.weight.set_value(paddle.randn(block.mlp.c_fc.weight.shape) * fc_std)
            block.mlp.c_proj.weight.set_value(paddle.randn(block.mlp.c_proj.weight.shape) * proj_std)

        if self.text_projection is not None:
            self.text_projection.set_value(paddle.randn(self.text_projection.shape) * (self.transformer.width ** -0.5))

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        assert unlocked_layers == 0 and freeze_layer_norm
        print(f'Freeze the text encoder', flush=True)
        for p in self.parameters():
            p.trainable = False

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # paddle uses additive attention mask; fill with -inf
        mask = paddle.full((self.num_pos, self.num_pos), float("-inf"))
        mask = paddle.triu(mask, 1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, [1, 0, 0, 0], value=1.0)
        additive_mask = paddle.full(cls_mask.shape, 0, dtype=cast_dtype)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = paddle.tile(additive_mask, (self.heads, 1, 1))
        return additive_mask

    def _repeat(self, t, N):
        return paddle.tile(t.unsqueeze(0).unsqueeze(0), [N, 1, 1])

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text)
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = paddle.concat([x, self._repeat(self.cls_emb, x.shape[0])], axis=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask.unsqueeze(0)[:seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = x.transpose((1, 0, 2))  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.transpose((1, 0, 2))  # LND -> NLD

        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[paddle.arange(x.shape[0]), text.argmax(axis=-1)], x

        if self.text_projection is not None:
            pooled = paddle.matmul(pooled, self.text_projection)

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(nn.Layer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        context_length: int = 77,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: callable = nn.GELU,
        norm_layer: callable = LayerNorm,
        output_dim: int = 512,
    ):
        super().__init__()
        self.context_length = context_length
        self.cross_attn = nn.LayerList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = self.create_parameter((width, output_dim), default_initializer=Normal())

        self.init_parameters()

    def init_parameters(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.cross_attn:
            block.attn.in_proj_weight.set_value(paddle.randn(block.attn.in_proj_weight.shape) * attn_std)
            block.attn.out_proj.weight.set_value(paddle.randn(block.attn.out_proj.weight.shape) * proj_std)
            block.mlp.c_fc.weight.set_value(paddle.randn(block.mlp.c_fc.weight.shape) * fc_std)
            block.mlp.c_proj.weight.set_value(paddle.randn(block.mlp.c_proj.weight.shape) * proj_std)

        if self.text_projection is not None:
            self.text_projection.set_value(paddle.randn(self.text_projection.shape) * (self.width ** -0.5))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # paddle uses additive attention mask; fill with -inf
        mask = paddle.full((self.context_length, self.context_length), float("-inf"))
        mask = paddle.triu(mask, 1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.transpose((1, 0, 2))  # NLD -> LND
        image_embs = image_embs.transpose((1, 0, 2))  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not paddle.in_static_mode():
                text_embs = paddle.checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = paddle.checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.transpose((1, 0, 2))  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = paddle.matmul(x, self.text_projection)

        return x

    @paddle.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable