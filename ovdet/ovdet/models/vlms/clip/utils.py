from typing import Union, List
import paddle
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> paddle.Tensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The maximum length to use; all CLIP models use 77 as the maximum length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the maximum length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder[""]
    eot_token = _tokenizer.encoder[""]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = paddle.zeros([len(all_tokens), context_length], dtype='int64')

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = paddle.to_tensor(tokens, dtype='int64')

    return result

def tokenize_dynamic(texts, context_length: int = 77, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder[""]
    eot_token = _tokenizer.encoder[""]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    lengths = [len(tokens) for tokens in all_tokens]
    context_length = min(context_length, max(lengths))
    result = paddle.zeros([len(all_tokens), context_length], dtype='int64')

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = paddle.to_tensor(tokens, dtype='int64')

    return result
