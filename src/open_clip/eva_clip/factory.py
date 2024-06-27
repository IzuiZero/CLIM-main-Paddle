import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import paddle

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import CLIP, CustomCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict, get_cast_dtype
from .openai import load_openai_model
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model
from .transform import image_transform
from .tokenizer import HFTokenizer, tokenize
from .utils import resize_clip_pos_embed, resize_evaclip_pos_embed, resize_visual_pos_embed, resize_eva_pos_embed

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, "r", encoding="utf8") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = dict(sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0])))

_rescan_model_configs()  # initial populate of model config registry

def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())

def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

def get_tokenizer(model_name):
    config = get_model_config(model_name)
    tokenizer = HFTokenizer(config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
    return tokenizer

# loading openai CLIP weights when is_openai=True for training
def load_state_dict(checkpoint_path: str, map_location: str='cpu', model_key: str='model|module|state_dict', is_openai: bool=False, skip_list: list=[]):
    if is_openai:
        model = paddle.jit.load(checkpoint_path)
        state_dict = model.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        checkpoint = paddle.load(checkpoint_path)
        for mk in model_key.split('|'):
            if isinstance(checkpoint, dict) and mk in checkpoint:
                state_dict = checkpoint[mk]
                break
            else:
                state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    for k in skip_list:
        if k in list(state_dict.keys()):
            logging.info(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]

    if os.getenv('RoPE') == '1':
        for k in list(state_dict.keys()):
            if 'freqs_cos' in k or 'freqs_sin' in k:
                del state_dict[k]
    return state_dict

def load_checkpoint(model, checkpoint_path, model_key="model|module|state_dict", strict=True):
    state_dict = load_state_dict(checkpoint_path, model_key=model_key, is_openai=False)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    if 'text.logit_scale' in state_dict and hasattr(model, 'logit_scale'):
        state_dict['logit_scale'] = state_dict['text.logit_scale']
        del state_dict['text.logit_scale']

    # resize_clip_pos_embed for CLIP and open CLIP
    if 'visual.positional_embedding' in state_dict:
        resize_clip_pos_embed(state_dict, model)
    # specified to eva_vit_model
    elif 'visual.pos_embed' in state_dict:
        resize_evaclip_pos_embed(state_dict, model)

    # resize_clip_pos_embed(state_dict, model)
    incompatible_keys = model.set_state_dict(state_dict, use_structured_name=strict)
    logging.info(f"incompatible_keys.missing_keys: {incompatible_keys.missing_keys}")
    return incompatible_keys

def load_clip_visual_state_dict(checkpoint_path: str, map_location: str='cpu', is_openai: bool=False, skip_list:list=[]):
    state_dict = load_state_dict(checkpoint_path, map_location=map_location, is_openai=is_openai, skip_list=skip_list)

    for k in list(state_dict.keys()):
        if not k.startswith('visual.'):
            del state_dict[k]
    for k in list(state_dict.keys()):
        if k.startswith('visual.'):
            new_k = k[7:]
            state_dict[new_k] = state_dict[k]
            del state_dict[k]
    return state_dict

def load_clip_text_state_dict(checkpoint_path: str, map_location: str='cpu', is_openai: bool=False, skip_list:list=[]):
    state_dict = load_state_dict(checkpoint_path, map_location=map_location, is_openai=is_openai, skip_list=skip_list)

    for k in list(state_dict.keys()):
        if k.startswith('visual.'):
            del state_dict[k]
    return state_dict

def get_pretrained_tag(pretrained_model):
    pretrained_model = pretrained_model.lower()
    if "laion" in pretrained_model or "open_clip" in pretrained_model:
        return "open_clip"
    elif "openai" in pretrained_model:
        return "clip"
    elif "eva" in pretrained_model and "clip" in pretrained_model:
        return "eva_clip"
    else:
        return "other"

def load_pretrained_checkpoint(
        model,
        visual_checkpoint_path,
        text_checkpoint_path,
        strict=True,
        visual_model=None,
        text_model=None,
        model_key="model|module|state_dict",
        skip_list=[]):
    visual_tag = get_pretrained_tag(visual_model)
    text_tag = get_pretrained_tag(text_model)

    logging.info(f"num of model state_dict keys: {len(model.state_dict().keys())}")
    visual_incompatible_keys, text_incompatible_keys = None, None
    if visual_checkpoint_path:
        if visual_tag == "eva_clip" or visual_tag == "open_clip":
            visual_state_dict = load_clip_visual_state_dict(visual_checkpoint_path, is_openai=False, skip_list=skip_list)
        elif visual_tag == "clip":
            visual_state_dict = load_clip_visual_state_dict(visual_checkpoint_path, is_openai=True, skip_list=skip_list)
        else:
            visual_state_dict = load_state_dict(visual_checkpoint_path, model_key=model_key, is_openai=False, skip_list=skip_list)
    
        # resize_clip_pos_embed for CLIP and open CLIP
        if 'positional_embedding' in visual_state_dict:
            resize_visual_pos_embed(visual_state_dict, model)
        # specified to EVA model
        elif 'pos_embed' in visual_state_dict:
            resize_eva_pos_embed(visual_state_dict, model)

        visual_incompatible_keys = model.visual.set_state_dict(visual_state_dict, use_structured_name=strict)
        logging.info(f"num of loaded visual_state_dict keys: {len(visual_state_dict.keys())}")
        logging.info(f"visual_incompatible_keys.missing_keys: {visual_incompatible_keys.missing_keys}")

    if text_checkpoint_path:
        if text_tag == "eva_clip" or text_tag == "open_clip":
            text_state_dict = load_clip_text_state_dict(text_checkpoint_path, is_openai=False, skip_list=skip_list)
        elif text_tag == "clip":
            text_state_dict = load_clip_text_state_dict(text_checkpoint_path, is_openai=True, skip_list=skip_list)
        else:
            text_state_dict = load_state_dict(visual_checkpoint_path, model_key=model_key, is_openai=False, skip_list=skip_list)

        text_incompatible_keys = model.text.set_state_dict(text_state_dict, use_structured_name=strict)
        
        logging.info(f"num of loaded text_state_dict keys: {len(text_state_dict.keys())}")
        logging.info(f"text_incompatible_keys.missing_keys: {text_incompatible_keys.missing_keys}")

    return visual_incompatible_keys, text_incompatible_keys

def create_model(
        model_name: str,
        pretrained: Optional[Union[str, bool]] = False,
        precision: Optional[str] = None,
        force_custom_clip: bool = False,
        cast_dtype: Optional[str] = None,
        pretrained_image: Optional[str] = None,
        pretrained_text: Optional[str] = None,
        **kwargs):
    logging.info(f"create model {model_name} ...")
    cast_dtype = get_cast_dtype(precision, cast_dtype)
    pretrained = pretrained or pretrained_image or pretrained_text

    # determine model config and if pretrained model_cfg should be used
    pretrained_cfg = get_pretrained_cfg(model_name)
    model_cfg = None
    if not is_pretrained_cfg(model_name):
        model_cfg = get_model_config(model_name)
    elif pretrained_cfg and model_name in pretrained_cfg:
        model_cfg = pretrained_cfg[model_name].get('model_cfg', None)
        if model_cfg is None:
            model_cfg = get_model_config(model_name)
    else:
        model_cfg = get_model_config(model_name)

    if model_cfg is None:
        raise RuntimeError(f"Model config for {model_name} not found.")

    # process model kwargs
    for k, v in kwargs.items():
        if k in model_cfg:
            logging.info(f'Overriding model config value {k}: {model_cfg[k]} -> {v}')
            model_cfg[k] = v

    if force_custom_clip or model_cfg.get('custom_text', False):
        model = CustomCLIP(**model_cfg)
    else:
        model = CLIP(**model_cfg)
    
    if cast_dtype is not None:
        convert_weights_to_lp(model, cast_dtype)

    if pretrained:
        # NOTE: by default, pretrained checkpoint loaded only for the single combined model case
        # the typical case for pretrained_image and pretrained_text args is separate eva model
        pretrained_models = []
        pretrained_tag = get_pretrained_tag(str(pretrained))
        if pretrained_tag == "other":
            pretrained_models.append(str(pretrained))
        if pretrained_image:
            pretrained_models.append(pretrained_image)
        if pretrained_text:
            pretrained_models.append(pretrained_text)

        pretrained_ckpt_path = []
        pretrained_image_path = pretrained_image
        pretrained_text_path = pretrained_text
        for pretrained_model in pretrained_models:
            ckpt_path = download_pretrained(pretrained_model, load_as=None, cache_dir=None, force_reload=False)
            pretrained_ckpt_path.append(ckpt_path)
        if pretrained_tag != "other" and len(pretrained_ckpt_path) > 0:
            pretrained_image_path = pretrained_ckpt_path[0]
            if len(pretrained_ckpt_path) > 1:
                pretrained_text_path = pretrained_ckpt_path[1]
            else:
                pretrained_text_path = pretrained_ckpt_path[0]

        if pretrained_image_path is not None or pretrained_text_path is not None:
            load_pretrained_checkpoint(
                model,
                visual_checkpoint_path=pretrained_image_path,
                text_checkpoint_path=pretrained_text_path,
                strict=False,
                visual_model=pretrained_image,
                text_model=pretrained_text)

    return model

def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[Union[str, bool]] = False,
        precision: Optional[str] = None,
        force_custom_clip: bool = False,
        cast_dtype: Optional[str] = None,
        pretrained_image: Optional[str] = None,
        pretrained_text: Optional[str] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs):
    model = create_model(
        model_name,
        pretrained=pretrained,
        precision=precision,
        force_custom_clip=force_custom_clip,
        cast_dtype=cast_dtype,
        pretrained_image=pretrained_image,
        pretrained_text=pretrained_text,
        **kwargs)

    model_cfg = model.visual.image_size if image_size is None else image_size
    if isinstance(model_cfg, int):
        img_size = (model_cfg, model_cfg)
    else:
        img_size = tuple(model_cfg)

    mean = OPENAI_DATASET_MEAN if image_mean is None else image_mean
    std = OPENAI_DATASET_STD if image_std is None else image_std
    transform = image_transform(img_size, is_train=False, mean=mean, std=std)

    return model, transform

def get_cast_dtype(precision, cast_dtype=None):
    return 'float16' if precision == 'fp16' else cast_dtype
