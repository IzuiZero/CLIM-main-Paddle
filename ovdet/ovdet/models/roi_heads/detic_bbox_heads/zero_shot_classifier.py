import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal

from mmdet.registry import MODELS


@MODELS.register_module()
class ZeroShotClassifier(nn.Layer):

    def __init__(
        self,
        in_features: int,
        out_features: int,  # num_classes
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        num_classes = out_features
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = self.create_parameter(shape=[1],
                                                  default_initializer=Normal(mean=use_bias, std=0.01))

        self.linear = nn.Linear(in_features, zs_weight_dim)

        if zs_weight_path == 'rand':
            zs_weight = paddle.randn((zs_weight_dim, num_classes))
            paddle.nn.initializer.Normal(std=0.01)(zs_weight)
        else:
            if zs_weight_path.endswith('npy'):
                zs_weight = paddle.to_tensor(
                    np.load(zs_weight_path),
                    dtype='float32').transpose([1, 0]).contiguous()  # D x C
            else:
                zs_weight = paddle.load(
                    zs_weight_path).astype('float32').transpose([1, 0]).contiguous()  # D x C
        zs_weight = paddle.concat(
            [zs_weight, paddle.zeros((zs_weight_dim, 1))], axis=1)  # D x (C + 1)

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, axis=0)

        if zs_weight_path == 'rand':
            self.zs_weight = self.create_parameter(shape=zs_weight.shape,
                                                   default_initializer=lambda _: zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        if classifier is not None:
            zs_weight = classifier.transpose([1, 0]).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, axis=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, axis=1)
        x = paddle.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x
