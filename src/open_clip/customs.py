from paddle import Tensor
from paddle.nn import MultiHeadAttention
from paddle.nn import functional as F
from typing import Optional, Tuple


class MultiheadSelfAttention(MultiHeadAttention):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, return_tokens: bool = False) \
            -> Tuple[Tensor, Tensor]:
        assert query is value and value is key       # self-attention
        if return_tokens:
            # in_projection
            tokens = F.linear(x=value, weight=self.q_proj.weight, bias=self.q_proj.bias)[..., -self.embed_dim:]
            # out_projection
            tokens = F.linear(x=tokens, weight=self.out_proj.weight, bias=self.out_proj.bias)
        else:
            tokens = None

        attn_output, attn_output_weights = self._multi_head_attention(
            query=query, key=key, value=value,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            q_proj_bias=self.q_proj.bias,
            k_proj_bias=self.k_proj.bias,
            v_proj_bias=self.v_proj.bias,
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)

        return attn_output, tokens  # , attn_output_weights

    def _multi_head_attention(self, query, key, value, embed_dim, num_heads, q_proj_weight, k_proj_weight, v_proj_weight,
                              q_proj_bias, k_proj_bias, v_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p,
                              out_proj_weight, out_proj_bias, training, key_padding_mask, need_weights, attn_mask):
        # PaddlePaddle's MultiHeadAttention forward implementation
        q = F.linear(query, weight=q_proj_weight, bias=q_proj_bias)
        k = F.linear(key, weight=k_proj_weight, bias=k_proj_bias)
        v = F.linear(value, weight=v_proj_weight, bias=v_proj_bias)

        q = self.transpose_multi_head(q, num_heads)
        k = self.transpose_multi_head(k, num_heads)
        v = self.transpose_multi_head(v, num_heads)

        attn_output, attn_output_weights = F.multi_head_attention(
            q, k, v,
            num_heads=num_heads,
            dropout=dropout_p,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights
        )

        attn_output = self.transpose_multi_head(attn_output, num_heads, reverse=True)
        attn_output = F.linear(attn_output, weight=out_proj_weight, bias=out_proj_bias)

        return attn_output, attn_output_weights

    @staticmethod
    def transpose_multi_head(x, num_heads, reverse=False):
        if reverse:
            # Reverse the multi-head transposition
            return x.transpose((0, 2, 1, 3)).reshape((x.shape[0], x.shape[1], -1))
        else:
            # Perform multi-head transposition
            return x.reshape((x.shape[0], x.shape[1], num_heads, -1)).transpose((0, 2, 1, 3))
