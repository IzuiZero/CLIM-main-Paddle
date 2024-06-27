import paddle

def get_autocast(precision):
    if precision == 'amp':
        return paddle.amp.auto_cast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # PaddlePaddle's equivalent of torch.bfloat16 is paddle.float16
        return lambda: paddle.amp.auto_cast(dtype=paddle.float16)
    else:
        return suppress  # Assuming suppress from contextlib is imported globally
