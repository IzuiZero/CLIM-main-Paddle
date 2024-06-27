from typing import Optional

import paddle
from paddle import nn
from paddle.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower

try:
    from paddlenlp.transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[paddle.dtype] = None,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (paddle.float16, paddle.bfloat16) else LayerNorm
    )

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class CoCa(nn.Layer):
    def __init__(
            self,
            embed_dim,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[paddle.dtype] = None,
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.logit_scale = self.create_parameter([1], default_initializer=nn.initializer.Constant(np.log(1 / 0.07)))
        self.pad_id = pad_id

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, axis=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize=True, embed_cls=True):
        text = text[:, :-1] if embed_cls else text # make space for CLS token
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, axis=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize=True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize=True, embed_cls=True):
        text_latent, _ = self._encode_text(text, normalize=normalize, embed_cls=embed_cls)
        return text_latent

    def forward(self, image, text, embed_cls=True, image_latent=None, image_embs=None):
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        labels = text[:, -token_embs.shape[1]:]

        logits = self.text_decoder(image_embs, token_embs)
        return {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": paddle.exp(self.logit_scale)
        }

    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with paddle.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.place

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    image_inputs = image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    return paddle.concat(
                        (output, paddle.ones((output.shape[0], seq_len-output.shape[1]), device=device, dtype=output.dtype) * self.pad_id),
                        axis=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_latent, image_embs = self._encode_image(image)

            if text is None:
                text = paddle.ones((image.shape[0], 1), device=device, dtype=paddle.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 2:
                text = text.unsqueeze(1)  # expand dims to be [batch, #num_captions, seq_len]

            generated = text
            for _ in range(seq_len):
                text_latent, token_embs = self._encode_text(generated[:, :, -max_seq_len:], embed_cls=False)
                logits = self.text_decoder(image_embs.unsqueeze(1), token_embs).squeeze(2)[:, -1]
                logits = logits / (temperature if temperature > 0 else 1.0)
                logits = logit_processor(generated, logits)
                logits = logit_warper(generated, logits)
                probs = F.softmax(logits, axis=-1)
                next_token = paddle.multinomial(probs, num_samples=1).squeeze(-1)

                generated = paddle.concat((generated, next_token.unsqueeze(-1)), axis=-1)

                # Greedy search if using top_p and top_k
                if logit_warper and eos_token_id in next_token:
                    break

                # Stop if all batches predict the eos_token_id
                if not logit_warper and paddle.all(next_token == eos_token_id).item():
                    break

            output_text = generated.squeeze(1) if num_dims == 2 else generated

            if fixed_output_length and output_text.shape[1] < seq_len:
                return paddle.concat(
                    (output_text, paddle.ones((output_text.shape[0], seq_len-output_text.shape[1]), device=device, dtype=output_text.dtype) * self.pad_id),
                    axis=1
                )
            return output_text

    def _generate_beamsearch(
        self,
        image_inputs,
        pad_token_id,
        eos_token_id,
        sot_token_id,
        num_beams,
        num_beam_groups,
        min_seq_len,
        stopping_criteria,
        logit_processor,
    ):
        batch_size = image_inputs.shape[0]
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            device=image_inputs.place,
            length_penalty=1.0,
            do_early_stopping=True,
            num_beam_hyps_to_keep=1,
        )

        # init beam search values
        beam_scores = paddle.zeros((batch_size, num_beams), dtype=paddle.float32, device=image_inputs.place)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.reshape((-1,))

        # encoder forward pass
        image_latent, image_embs = self._encode_image(image_inputs)
        image_embs = image_embs.unsqueeze(1).tile(repeat_times=[1, num_beams, 1, 1])
        image_latent = image_latent.tile(repeat_times=[num_beams, 1])

        input_ids = paddle.ones((batch_size * num_beams, 1), device=image_inputs.place, dtype=paddle.long) * sot_token_id
        cur_len = 1

        # while loop
        while cur_len < stopping_criteria.max_length:
            text_latent, token_embs = self._encode_text(input_ids[:, -min_seq_len:], embed_cls=False)
            logits = self.text_decoder(image_embs, token_embs).squeeze(2)[:, -1]
            next_token_scores = F.log_softmax(logits, axis=-1)
            next_token_scores = logit_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            next_token_scores = next_token_scores.reshape((batch_size, num_beams * next_token_scores.shape[1]))

            next_tokens = paddle.topk(next_token_scores, 2 * num_beams, axis=1).indices
            next_indices = paddle.topk(next_token_scores, 2 * num_beams, axis=1).indices

            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs['next_beam_scores']
            beam_next_tokens = beam_outputs['next_beam_tokens']
            beam_idx = beam_outputs['next_beam_indices']

            input_ids = paddle.concat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], axis=-1
            )

            if beam_scorer.is_done:
                break
            cur_len += 1

        output = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return output['sequences']
