import re
import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.nn import functional as F
from paddle.tensor import Tensor as TensorType

try:
    from paddlenlp.transformers import AutoModel, AutoTokenizer, PretrainedConfig
    from paddlenlp.transformers.model_utils import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    paddlenlp = None

from .hf_configs import arch_dict

# Utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

# Pooler registration
_POOLERS = {}

def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls

@register_pooler
class MeanPooler(nn.Layer):
    """Mean pooling"""
    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return paddle.sum(masked_output, axis=1) / paddle.sum(attention_mask, axis=-1, keepdim=True)

@register_pooler
class MaxPooler(nn.Layer):
    """Max pooling"""
    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -float('inf'))
        return paddle.max(masked_output, axis=1).values

@register_pooler
class ClsPooler(nn.Layer):
    """CLS token pooling"""
    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        if (self.use_pooler_output and 
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)):
            return x.pooler_output
        
        return x.last_hidden_state[:, self.cls_token_position, :]

class HFTextEncoder(nn.Layer):
    """HuggingFace model adapter"""
    def __init__(
            self, 
            model_name_or_path: str,
            output_dim: int,
            tokenizer_name: str = None,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj: str = None,
            pretrained: bool = True,
            masked_language_modeling: bool = False):
        super().__init__()

        self.output_dim = output_dim
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if paddlenlp is None:
            raise RuntimeError("Please `pip install paddlenlp` to use pre-trained PaddleNLP models")
        
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            if masked_language_modeling:
                create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (
                    AutoModel.from_config, self.config)
            else:
                create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (
                    AutoModel.from_config, self.config)
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            if masked_language_modeling:
                self.transformer = AutoModel.from_config(config)
            else:
                self.transformer = AutoModel.from_config(config)

        if pooler_type is None:
            self.pooler = _POOLERS[(arch_dict[self.config.model_type]["pooler"])]()
        else:
            self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj is None):
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias_attr=False)
        elif proj == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias_attr=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias_attr=False),
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = paddle.randint(0, 2, input_ids.shape).astype('bool')

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        indices_replaced = paddle.logical_and(
            paddle.randint(0, 2, input_ids.shape).astype('bool'), masked_indices)
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = paddle.logical_and(
            paddle.randint(0, 2, input_ids.shape).astype('bool'),
            masked_indices.logical_and(~indices_replaced))
        random_words = paddle.randint(0, vocab_size, input_ids.shape).astype('int64')
        input_ids[indices_random] = random_words[indices_random]

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward_mlm(self, input_ids, image_embeds, mlm_probability=0.25):
        labels = input_ids.clone()
        attn_mask = (input_ids != self.config.pad_token_id).astype('int64')
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype='int64')
        vocab_size = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["vocab_size"])
        probability_matrix = paddle.full(labels.shape, mlm_probability)
        input_ids, labels = self.mask(input_ids, vocab_size, input_ids.device, targets=labels,
                                      masked_indices=None, probability_matrix=probability_matrix)
        mlm_output = self.transformer(input_ids,
                                      attention_mask=attn_mask,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=image_atts,
                                      labels=labels)
        return mlm_output.loss

    def forward(self, x: Tensor) -> Tensor:
        attn_mask = (x != self.config.pad_token_id).astype('int64')
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)

        return self.proj(pooled_out)

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:
            for n, p in self.transformer.named_parameters():
                p.trainable = not freeze_layer_norm if "LayerNorm" in n.split(".") else False
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(self.transformer, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        for module in modules:
            for n, p in module.named_parameters():
                p.trainable = not freeze_layer_norm if "LayerNorm" in n.split(".") else False

    def get_num_layers(self):
        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        return len(layer_list)

    def init_parameters(self):
        pass
