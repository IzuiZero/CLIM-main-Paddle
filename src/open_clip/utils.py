import collections.abc
from paddle import nn
from paddle.static import InputSpec

class FrozenBatchNorm2d(nn.Layer):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.affine = False  # FrozenBatchNorm2d does not have trainable parameters
        self.eps = 1e-5  # Default epsilon value in FrozenBatchNorm2d
        self.weight = self.create_parameter(shape=[num_features], default_initializer=None, is_bias=False)
        self.bias = self.create_parameter(shape=[num_features], default_initializer=None, is_bias=False)
        self.running_mean = self.create_parameter(shape=[num_features], default_initializer=None, is_bias=False)
        self.running_var = self.create_parameter(shape=[num_features], default_initializer=None, is_bias=False)

    def forward(self, x):
        scale = self.weight / (self.running_var + self.eps).sqrt()
        bias = self.bias - self.running_mean * scale
        return x * scale.reshape([1, -1, 1, 1]) + bias.reshape([1, -1, 1, 1])


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of `BatchNorm2d`, it is converted into `FrozenBatchNorm2d` and returned.
    Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (paddle.nn.Layer): Any PaddlePaddle module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        paddle.nn.Layer: Resulting module
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.BatchNorm2D)):
        res = FrozenBatchNorm2d(module._variance._shape[0])
        res.num_features = module._variance._shape[0]
        res.affine = module._parameters is not None
        if module._parameters is not None:
            res.weight = module.weight.clone().detach()
            res.bias = module.bias.clone().detach()
        res.running_mean = module._mean.clone().detach()
        res.running_var = module._variance.clone().detach()
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_sublayer(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)
