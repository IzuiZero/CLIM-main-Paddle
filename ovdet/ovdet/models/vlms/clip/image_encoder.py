import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.logging import print_log
from collections import OrderedDict
from .common import Transformer, LayerNorm  # Assuming common.py contains the Transformer and LayerNorm classes


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.avgpool = nn.AvgPool2D(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.AvgPool2D(stride),
                nn.Conv2D(inplanes, planes * self.expansion, 1, stride=1, bias_attr=False),
                nn.BatchNorm2D(planes * self.expansion)
            )

    def forward(self, x: paddle.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Layer):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = self.create_parameter(
            shape=[spacial_dim ** 2 + 1, embed_dim],
            default_initializer=Normal(0, embed_dim ** -0.5),
            dtype='float32'
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_tokens=False, attn_masks=None):
        N, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).transpose([2, 0, 1])  # NCHW -> (HW)NC
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding.unsqueeze(1)  # (HW+1)NC

        if return_tokens:
            tokens = self.c_proj(
                self.v_proj(x[1:])).transpose([1, 2, 0]).reshape([N, -1, H, W])
        else:
            tokens = None

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=paddle.concat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            attn_mask=attn_masks
        )

        return x[0], tokens


# @MODELS.register_module()
class CLIPResLayer4(BaseModule):
    def __init__(self, inplanes, planes, blocks, stride=1, freeze=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze = freeze
        layers = [Bottleneck(inplanes, planes, stride)]

        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))

        self.layer4 = nn.Sequential(*layers)

    def init_weights(self):
        super().init_weights()
        if self.freeze:
            print_log('Freeze the weights of CLIPResLayer4.')
            self.eval()
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, x):
        return self.layer4(x)


# @MODELS.register_module()
class CLIPResNet(BaseModule):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64,
                 freeze=True, **kwargs
                 ):
        super().__init__(**kwargs)
        self.freeze = freeze
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.attn_resolution = input_resolution // 32

        # the 3-layer stem
        self.conv1 = nn.Conv2D(3, width // 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.conv2 = nn.Conv2D(width // 2, width // 2, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.conv3 = nn.Conv2D(width // 2, width, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)
        self.avgpool = nn.AvgPool2D(2)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.num_heads = heads
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def init_weights(self):
        super().init_weights()
        if self.freeze:
            print_log('Freeze the weights of CLIPResNet.')
            self.eval()
            for param in self.parameters():
                param.stop_gradient = True

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def encode_image(self, x: paddle.Tensor, normalize=True, return_tokens=False, attn_masks=None):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.astype(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x, image_tokens = self.attnpool(x, return_tokens=return_tokens, attn_masks=attn_masks)
        if normalize:
            x = F.normalize(x, p=2, axis=-1)
        if return_tokens:
            assert image_tokens is not None
            return x, image_tokens
        else:
            return x


# @MODELS.register_module()
class CLIPViT(BaseModule):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 freeze=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze = freeze
        self.input_resolution = input_resolution
        self.attn_resolution = input_resolution // patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias_attr=False)

        scale = width ** -0.5
        self.class_embedding = self.create_parameter(
            shape=[width],
            default_initializer=Normal(0, scale),
            dtype='float32'
        )
        self.positional_embedding = self.create_parameter(
            shape=[(input_resolution // patch_size) ** 2 + 1, width],
            default_initializer=Normal(0, scale),
            dtype='float32'
        )
        self.ln_pre = LayerNorm(width)
        self.num_heads = heads

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = self.create_parameter(
            shape=[width, output_dim],
            default_initializer=Normal(0, scale),
            dtype='float32'
        ) if output_dim is not None else None

    def init_weights(self):
        super().init_weights()
        if self.freeze:
            print_log('Freeze the weights of CLIPViT.')
            self.eval()
            for param in self.parameters():
                param.stop_gradient = True

    def rescale_positional_embedding(self, out_size, dtype):
        rescaled_positional_embedding = paddle.zeros([1 + out_size ** 2, self.positional_embedding.shape[1]], dtype=dtype)
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].transpose([1, 0]).reshape([self.pe_grid_size, self.pe_grid_size, -1])
        pe_2d = F.interpolate(pe_2d, (out_size, out_size), mode='bilinear').reshape([-1, out_size ** 2]).transpose([1, 0])
        rescaled_positional_embedding[1:] = pe_2d

        return rescaled_positional_embedding.astype(dtype)

    def encode_image(self, x: paddle.Tensor, normalize=True, return_tokens=False, attn_masks=None):
        x = self.conv1(x)
        grid_size = x.shape[-1]
        x = x.reshape([x.shape[0], x.shape[1], -1]).transpose([0, 2, 1])
        x = paddle.concat([self.class_embedding.unsqueeze(0).tile([x.shape[0], 1, 1]),
                           x], axis=1)
        if grid_size == self.attn_resolution:
            pe = self.positional_embedding
        else:
            pe = self.rescale_positional_embedding(out_size=grid_size, dtype=x.dtype)
        x = x + pe
        x = self.ln_pre(x)

        x = x.transpose([1, 0, 2])
        x, image_tokens = self.transformer(x, return_tokens=return_tokens, cls_indices=0,
                                           attn_masks=attn_masks)
        x = x.transpose([1, 0, 2])

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x.matmul(self.proj)

        if normalize:
            x = F.normalize(x, p=2, axis=-1)

        if return_tokens:
            image_tokens = image_tokens.transpose([1, 0, 2])
            image_tokens = self.ln_post(image_tokens)
            if self.proj is not None:
                image_tokens = image_tokens.matmul(self.proj)

            image_tokens = image_tokens[:, 1:].transpose([0, 2, 1]).reshape([x.shape[0], -1, grid_size, grid_size])
            return x, image_tokens
        else:
            assert image_tokens is None
            return x
