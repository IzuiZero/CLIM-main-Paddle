import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer.transformer import TransformerEncoderLayer, TransformerEncoder
from functools import partial
from paddle.nn.layer.norm import LayerNorm
from ovdet.utils import multi_apply
from .simple_tokenizer import SimpleTokenizer
from mmengine.logging import print_log
from mmdet.registry import MODELS

class Transformer(nn.Layer):
    def __init__(self, width, layers, heads, attn_mask=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        encoder_layer = TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=layers)
        self.attn_mask = attn_mask

    def forward(self, x, return_tokens=False, cls_indices=None, attn_masks=None):
        if attn_masks is None:
            attn_masks = self.attn_mask
        if attn_masks is not None:
            attn_masks = attn_masks.unsqueeze([1, 2])  # for paddle
        output = self.transformer(x, src_mask=attn_masks)
        if return_tokens:
            return output, output
        return output, None


@MODELS.register_module()
class CLIPTextEncoder(nn.Layer):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 freeze=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze = freeze
        self.context_length = context_length

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
        self.positional_embedding = self.create_parameter(
            shape=[self.context_length, transformer_width],
            default_initializer=nn.initializer.Normal()
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = self.create_parameter(
            shape=[transformer_width, embed_dim],
            default_initializer=nn.initializer.Normal()
        )

    def init_weights(self):
        if self.freeze:
            print_log('Freeze the weights of CLIP text encoder.')
            for param in self.parameters():
                param.stop_gradient = True

    def build_attention_mask(self, context_length=None):
        if context_length is None:
            context_length = self.context_length
        mask = paddle.full((context_length, context_length), float("-inf"))
        mask = paddle.triu(mask, 1)
        return mask

    def encode_text(self, text, normalize=True, return_word_tokens=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        if text.shape[1] <= self.context_length:
            x = x + self.positional_embedding[:text.shape[1]]
            custom_attn_mask = None
        else:
            pe = self.positional_embedding
            new_pe = F.interpolate(pe.T.unsqueeze(0), size=text.shape[1], mode='linear', align_corners=True)[0].T
            custom_attn_mask = self.build_attention_mask(text.shape[1])
            x = x + new_pe
        x = x.transpose([1, 0, 2])  # NLD -> LND
        x, word_tokens = self.transformer(x, return_tokens=return_word_tokens,
                                          cls_indices=text.argmax(axis=-1),
                                          attn_masks=custom_attn_mask)
        x = x.transpose([1, 0, 2])  # LND -> NLD
        x = self.ln_final(x)

        out = paddle.matmul(x[paddle.arange(x.shape[0]), text.argmax(axis=-1)], self.text_projection)
        x = paddle.matmul(x, self.text_projection)
        if normalize:
            out = F.normalize(out, p=2, axis=-1)

        if return_word_tokens:
            word_tokens = word_tokens.transpose([1, 0, 2])  # LND -> NLD
            word_tokens = self.ln_final(word_tokens)
            word_tokens = paddle.matmul(word_tokens, self.text_projection)
            if normalize:
                word_tokens = F.normalize(word_tokens, axis=-1)
                x = F.normalize(x, axis=-1)
            word_tokens = [seq[:end_token_id]
                           for seq, end_token_id in zip(word_tokens, text.argmax(axis=-1))]
            return out, word_tokens, x
        else:
            assert word_tokens is None
            return out

    def encode_text_endk(self, text, stepk=12, normalize=True, **kwargs):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding[:text.shape[1]]
        x = x.transpose([1, 0, 2])  # NLD -> LND

        for i in range(stepk):
            x, _ = self.transformer.transformer.layers[i](x)

        out = x.transpose([1, 0, 2])  # LND -> NLD

        out = out[paddle.arange(out.shape[0]), text.argmax(axis=-1)]

        if normalize:
            out = F.normalize(out, axis=-1, p=2)

        return out, x, text.argmax(axis=-1)

    def encode_pseudo_text_endk(self, x, end_token_ids, text_pe=True,
                                stepk=12, normalize=True):
        if text_pe:
            x = x + self.positional_embedding[:x.shape[1]]
        else:
            for i in range(x.shape[0]):
                x[i, end_token_ids[i]:] = x[i, end_token_ids[i]:] + self.positional_embedding[end_token_ids[i]:]
                x[i, 0] = x[i, 0] + self.positional_embedding[0]

        x = x.transpose([1, 0, 2])  # NLD -> LND
        for i in range(stepk):
            x, _ = self.transformer.transformer.layers[i](x)

        out = x.transpose([1, 0, 2])  # LND -> NLD

        out = out[paddle.arange(out.shape[0]), end_token_ids]

        if normalize:
            out = F.normalize(out, axis=-1, p=2)

        return out, x, end_token_ids

    def encode_pseudo_text(self, x, end_token_ids, text_pe=True, normalize=True,
                           return_word_tokens=False):
        if text_pe:
            x = x + self.positional_embedding[:x.shape[1]]
        else:
            for i in range(x.shape[0]):
                x[i, end_token_ids[i]:] = x[i, end_token_ids[i]:] + self.positional_embedding[end_token_ids[i]:]
                x[i, 0] = x[i, 0] + self.positional_embedding[0]

        x = x.transpose([1, 0, 2])  # NLD -> LND

        num_steps = len(self.transformer.transformer.layers)
        for i in range(num_steps - 1):
            x, _ = self.transformer.transformer.layers[i](x)
        x, word_tokens = self.transformer.transformer.layers[-1](x, return_tokens=return_word_tokens,
                                                                 cls_indices=end_token_ids)
        x = x.transpose([1, 0, 2])  # LND -> NLD
        x = self.ln_final(x)

        out = paddle.matmul(x[paddle.arange(x.shape[0]), end_token_ids], self.text_projection)

        if normalize:
            out = F.normalize(out, axis=-1, p=2)
        if return_word_tokens:
            word_tokens = word_tokens.transpose([1, 0, 2])  # LND -> NLD
            word_tokens = self.ln_final(word_tokens)
            word_tokens = paddle.matmul(word_tokens, self.text_projection)
            word_tokens = [seq[:end_token_id]
                           for seq, end_token_id in zip(word_tokens, end_token_ids)]
            return out, word_tokens
        else:
            assert word_tokens is None
            return out

    def prepare_pseudo_text_tensor(self, pseudo_tokens, valid_mask):
        device = pseudo_tokens.place
        num_preds, num_words, word_dim = pseudo_tokens.shape
        sot_token = self.token_embedding(paddle.to_tensor([self.sot_token], place=device))
        eot_token = self.token_embedding(paddle.to_tensor([self.eot_token], place=device))
        sot_token = sot_token.reshape([1, 1, word_dim]).tile([num_preds, 1, 1])
        eot_token = eot_token.reshape([1, 1, word_dim]).tile([num_preds, 1, 1])
        pseudo_tokens = paddle.concat([sot_token, pseudo_tokens, eot_token], axis=1)
        num_words += 2
        assert valid_mask.shape == pseudo_tokens.shape[:2]
        pseudo_tokens_flat = pseudo_tokens.reshape([-1, word_dim])
        valid_mask_flat = valid_mask.reshape([-1])

        empty_token = self.token_embedding(paddle.to_tensor([0], place=device))
        template_flat = empty_token.reshape([1, word_dim]).tile([num_preds * num_words, 1])

        valid_mask_zero_pad = paddle.concat([paddle.zeros_like(valid_mask[:, :1]), valid_mask], axis=-1)
        pe_indices = (valid_mask_zero_pad > 0.0).cumsum(axis=-1)[:, :-1]
        pe_indices_flat = (pe_indices + (paddle.arange(num_preds, place=pe_indices.place) * num_words)[:, None]).reshape([-1])

        template_flat[pe_indices_flat[valid_mask_flat > 0.0]] = pseudo_tokens_flat[valid_mask_flat > 0.0]
        pseudo_tokens = template_flat.reshape([num_preds, num_words, word_dim])
        end_token_ids = (valid_mask > 0.0).sum(axis=-1).astype(paddle.int64) - 1

        return pseudo_tokens, end_token_ids

    def prepare_pseudo_text(self, pseudo_tokens, context_length):
        device = pseudo_tokens[0].place
        sot_token = self.token_embedding(paddle.to_tensor([self.sot_token], place=device))  # [batch_size, n_ctx, d_model]
        eot_token = self.token_embedding(paddle.to_tensor([self.eot_token], place=device))
        empty_token = self.token_embedding(paddle.to_tensor([0], place=device))
        pseudo_tokens = [paddle.concat([sot_token, tokens, eot_token], axis=0) for tokens in pseudo_tokens]

        def _pad_sequence(tokens):
            if tokens.shape[0] > context_length:
                x = tokens[list(range(context_length - 1)) + [tokens.shape[0] - 1]]
                end_token_id = context_length - 1
            else:
                x = paddle.concat([tokens, empty_token.tile([context_length - tokens.shape[0], 1])], axis=0)
                end_token_id = tokens.shape[0] - 1
            return x, end_token_id

        x, end_token_ids = multi_apply(_pad_sequence, pseudo_tokens)
        x = paddle.stack(x, axis=0)

        return x, paddle.to_tensor(end_token_ids, dtype=paddle.int64, place=x.place)
