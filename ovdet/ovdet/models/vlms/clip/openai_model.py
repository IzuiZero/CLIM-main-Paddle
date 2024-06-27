import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ovdet.utils.misc import multi_apply
import numpy as np
from .simple_tokenizer import SimpleTokenizer
from .common import LayerNorm, Transformer
from .image_encoder import Bottleneck, AttentionPool2d

class ModifiedResNet(nn.Layer):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

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
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.num_heads = heads
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_tokens=False, attn_masks=None):
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
        if return_tokens:
            assert image_tokens is not None
            return x, image_tokens
        else:
            return x


class VisionTransformer(nn.Layer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.pe_grid_size = input_resolution // patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias_attr=False)

        scale = width ** -0.5
        self.class_embedding = self.create_parameter(shape=[width], default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=scale))
        self.positional_embedding = self.create_parameter(shape=[(input_resolution // patch_size) ** 2 + 1, width], default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=scale))
        self.ln_pre = LayerNorm(width)
        self.num_heads = heads

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = self.create_parameter(shape=[width, output_dim], default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=scale))

    def rescale_positional_embedding(self, out_size, dtype):
        rescaled_positional_embedding = self.positional_embedding.new_zeros([1 + out_size ** 2, self.positional_embedding.shape[1]])
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].T.contiguous().view(
            1, -1, self.pe_grid_size, self.pe_grid_size)
        pe_2d = F.interpolate(pe_2d, (out_size, out_size), mode='bilinear').view(-1, out_size**2)
        rescaled_positional_embedding[1:] = pe_2d.T.contiguous()

        return rescaled_positional_embedding.astype(dtype)

    def forward(self, x, return_tokens, attn_masks=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid_size = x.shape[-1]
        x = x.reshape([x.shape[0], x.shape[1], -1])  # shape = [*, width, grid ** 2]
        x = x.transpose([0, 2, 1])  # shape = [*, grid ** 2, width]
        x = paddle.concat([self.class_embedding.astype(x.dtype)
                           + paddle.zeros([x.shape[0], 1, x.shape[-1]], dtype=x.dtype, device=x.device),
                           x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        if grid_size == self.pe_grid_size:
            pe = self.positional_embedding.astype(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=grid_size, dtype=x.dtype)
        x = x + pe
        x = self.ln_pre(x)

        x = x.transpose([1, 0, 2])  # NLD -> LND
        x, image_tokens = self.transformer(x, return_tokens=return_tokens, cls_indices=0,
                                           attn_masks=attn_masks)
        x = x.transpose([1, 0, 2])  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = paddle.matmul(x, self.proj)
        if return_tokens:
            image_tokens = image_tokens.transpose([1, 0, 2])
            image_tokens = self.ln_post(image_tokens)
            if self.proj is not None:
                image_tokens = paddle.matmul(image_tokens, self.proj)

            # return the processed image token embeddings
            image_tokens = image_tokens[:, 1:].transpose([0, 2, 1]).contiguous()
            image_tokens = image_tokens.reshape([x.shape[0], -1, grid_size, grid_size])
        else:
            assert image_tokens is None

        return x, image_tokens


class CLIP(nn.Layer):
    def __init__(self,
                 embed_dim: int,
                 state_file: str,
                 # vision
                 use_image_encoder: bool,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 use_text_encoder: bool,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.state_file = state_file
        self.context_length = context_length
        self.use_image_encoder = use_image_encoder
        self.use_text_encoder = use_text_encoder
        self.input_resolution = image_resolution
        if use_image_encoder:
            if isinstance(vision_layers, (tuple, list)):
                vision_heads = vision_width * 32 // 64
                self.visual = ModifiedResNet(
                    layers=vision_layers,
                    output_dim=embed_dim,
                    heads=vision_heads,
                    input_resolution=image_resolution,
                    width=vision_width
                )
            else:
                vision_heads = vision_width // 64
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim
                )
        else:
            self.visual = None
        if self.use_text_encoder:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.tokenizer = SimpleTokenizer()
            self.sot_token = self.tokenizer.encoder[""]
            self.eot_token = self.tokenizer.encoder[""]
            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Embedding(context_length, transformer_width)
        self.ln_final = LayerNorm(embed_dim)
        self.project = nn.Linear(embed_dim, embed_dim)

    def build_attention_mask(self):
        # build a lower triangular mask
        return paddle.tril(paddle.ones([self.context_length, self.context_length], dtype=paddle.int64))

    def forward(self, text, image):
        if self.use_text_encoder and self.use_image_encoder:
            # prepare the image tensor
            image = self.visual(image.astype(self.conv1.weight.dtype), return_tokens=False)
            # tokenize the text input
            text_tokens = self.tokenizer(text)
            # prepend the SOT token to the tokenized text
            text_tokens = paddle.concat([
                paddle.full_like(image[:, :1, :], self.sot_token),
                text_tokens
            ], axis=1)

            # obtain the token and vision embeddings
            token_embeddings = self.token_embedding(text_tokens[:, :self.context_length])
            # apply the positional embeddings
            positional_embeddings = self.positional_embedding(paddle.arange(self.context_length).astype(paddle.int64))
            # add the positional embeddings to the token embeddings
            x = token_embeddings + positional_embeddings
            # apply layer normalization
            x = self.ln_final(x)
            # apply the transformer to obtain the text embeddings
            x, _ = self.transformer(x, attn_mask=self.build_attention_mask())
            # apply layer normalization
            x = self.ln_final(x)
            # project the text embeddings to the embedding dimension
            x = self.project(x)
            # obtain the cosine similarity between the text and image embeddings
            return paddle.matmul(x, image) / (paddle.norm(x, axis=-1, keepdim=True) * paddle.norm(image, axis=-1, keepdim=True))

        # if using only text encoder
        elif self.use_text_encoder:
            # tokenize the text input
            text_tokens = self.tokenizer(text)
            # prepend the SOT token to the tokenized text
            text_tokens = paddle.concat([
                paddle.full_like(image[:, :1, :], self.sot_token),
                text_tokens
            ], axis=1)

            # obtain the token and vision embeddings
            token_embeddings = self.token_embedding(text_tokens[:, :self.context_length])
            # apply the positional embeddings
            positional_embeddings = self.positional_embedding(paddle.arange(self.context_length).astype(paddle.int64))
            # add the positional embeddings to the token embeddings
            x = token_embeddings + positional_embeddings
            # apply layer normalization
            x = self.ln_final(x)
            # apply the transformer to obtain the text embeddings
            x, _ = self.transformer(x, attn_mask=self.build_attention_mask())
            # apply layer normalization
            x = self.ln_final(x)
            # project the text embeddings to the embedding dimension
            x = self.project(x)
            # obtain the cosine similarity between the text and image embeddings
            return paddle.matmul(x, image) / (paddle.norm(x, axis=-1, keepdim=True) * paddle.norm(image, axis=-1, keepdim=True))

        # if using only image encoder
        elif self.use_image_encoder:
            # obtain the image embeddings
            image = self.visual(image.astype(self.conv1.weight.dtype), return_tokens=False)
            # apply layer normalization
            x = self.ln_final(image)
            # project the image embeddings to the embedding dimension
            x = self.project(x)
            return x

        raise ValueError("At least one of `use_text_encoder` or `use_image_encoder` must be True.")

    def tokenize(self, text: str):
        return paddle.full_like(text, self.tokenizer.encoder[text.lower()])

    def from_pretrained(self, state_dict):
        self.load_state_dict(state_dict)
