""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

import paddle

from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, paddle.CUDAPlace]] = None,
        jit: bool = True,
        cache_dir: Optional[str] = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, paddle.CUDAPlace]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : paddle.nn.Layer
        The CLIP model
    preprocess : Callable[[PIL.Image], paddle.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    if precision is None:
        precision = 'fp32' if device == paddle.CPUPlace() else 'fp16'

    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    try:
        # loading JIT archive
        model = paddle.jit.load(model_path)
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = paddle.load(model_path)

    if not jit:
        # Build a non-jit model from the OpenAI jitted model state dict
        cast_dtype = get_cast_dtype(precision)
        try:
            model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

        # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
        model = model.to(device)
        if precision.startswith('amp') or precision == 'fp32':
            model.float()
        elif precision == 'bf16':
            convert_weights_to_lp(model, dtype=paddle.bfloat16)

        return model

    # patch the device names
    device_holder = paddle.jit.trace(lambda: paddle.ones([]).to(device), [])
    device_node = [n for n in device_holder.graph.nodes() if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.forward.main_program] if hasattr(module, "forward") else []

        for graph in graphs:
            for node in graph.nodes():
                if "value" in node.attrs() and str(node.attr("value")).startswith("cuda"):
                    node.copy_attrs(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 (typically for CPU)
    if precision == 'fp32':
        float_holder = paddle.jit.trace(lambda: paddle.ones([]).astype('float32'), [])
        float_input = list(float_holder.graph.nodes())[1]
        float_node = float_input

        def patch_float(module):
            graphs = [module.forward.main_program] if hasattr(module, "forward") else []

            for graph in graphs:
                for node in graph.nodes():
                    if node.op_type == "cast" and node.attr("dtype") == 5:
                        node.copy_attrs(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model
