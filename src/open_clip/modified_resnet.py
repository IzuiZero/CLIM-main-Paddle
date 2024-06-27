import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import roi_align

class FrozenBatchNorm2d(nn.Layer):
    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = self.create_parameter(shape=[num_features], default_initializer=paddle.nn.initializer.Constant(1.0))
        self.bias = self.create_parameter(shape=[num_features], default_initializer=paddle.nn.initializer.Constant(0.0))
        self.running_mean = self.create_parameter(shape=[num_features], default_initializer=paddle.nn.initializer.Constant(0.0), trainable=False)
        self.running_var = self.create_parameter(shape=[num_features], default_initializer=paddle.nn.initializer.Constant(1.0 - eps), trainable=False)

    def forward(self, x):
        if self.training:
            scale = self.weight * paddle.rsqrt(self.running_var + self.eps)
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape((1, -1, 1, 1))
            bias = bias.reshape((1, -1, 1, 1))
            return x * scale.astype(x.dtype) + bias.astype(x.dtype)
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = paddle.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = paddle.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def extra_repr(self):
        return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.eps})"

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        bn_module = (nn.BatchNorm2D, nn.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.set_value(module.weight.clone().detach())
                res.bias.set_value(module.bias.clone().detach())
            res.running_mean.set_value(module.running_mean)
            res.running_var.set_value(module.running_var)
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_sublayer(name, new_child)
        return res


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.act2 = nn.ReLU()

        self.avgpool = nn.AvgPool2D(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.act3 = nn.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.AvgPool2D(stride),
                nn.Conv2D(inplanes, planes * self.expansion, kernel_size=1, stride=1, bias_attr=False),
                nn.BatchNorm2D(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Layer):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None,
                 freeze_output=True):
        super().__init__()
        self.positional_embedding = self.create_parameter(shape=[spacial_dim ** 2 + 1, embed_dim], 
                                                         default_initializer=paddle.nn.initializer.Normal(std=1.0) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spacial_dim = spacial_dim

        if freeze_output:
            print('Freeze the V2L layer', flush=True)
            for p in self.c_proj.parameters():
                p.trainable = False
            for p in self.v_proj.parameters():
                p.trainable = False

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).transpose([2, 0, 1])  # NCHW -> (HW)NC
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding.unsqueeze(1).astype(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=paddle.concat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

    def rescale_positional_embedding(self, out_size, dtype):
        h, w = out_size
        rescaled_positional_embedding = paddle.zeros((1 + h * w, self.positional_embedding.shape[1]), dtype=dtype)
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].transpose([1, 0]).reshape((self.spacial_dim, self.spacial_dim, -1))
        pe_2d = F.interpolate(pe_2d, out_size, mode='bicubic', align_corners=False).reshape((-1, h * w)).transpose([1, 0])
        rescaled_positional_embedding[1:] = pe_2d

        return rescaled_positional_embedding.astype(dtype)

    def proj_without_attn(self, value):
        value = F.linear(value, self.v_proj.weight, bias=self.v_proj.bias)
        value = F.linear(value, self.c_proj.weight, bias=self.c_proj.bias)

        return value

    def forward_dense(self, x):
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).transpose([2, 0, 1])  # NCHW -> (HW)NC
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)  # (HW+1)NC
        if h == self.spacial_dim and w == self.spacial_dim:
            pe = self.positional_embedding.unsqueeze(1).astype(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype).unsqueeze(1)

        x = x + pe  # (HW+1)NC

        x = self.proj_without_attn(x)

        return x[1:].transpose([1, 2, 0]).reshape((bs, -1, h, w))


class ModifiedResNet(nn.Layer):
    def __init__(self, layers, output_dim, heads, image_size=224, width=64,
                 freeze_output=True,
                 freeze_all_bns=True):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.freeze_output = freeze_output
        self.freeze_all_bns = freeze_all_bns
        # the 3-layer stem
        self.conv1 = nn.Conv2D(3, width // 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2D(width // 2, width // 2, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2D(width // 2, width, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)
        self.act3 = nn.ReLU()
        self.avgpool = nn.AvgPool2D(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim, freeze_output)
        self.attnpool_input_size = image_size // 32

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def lock(self, unlocked_groups=0, freeze_bn_stats=True):
        assert freeze_bn_stats
        def _lock(module):
            for param in module.parameters():
                param.trainable = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(module)
            module.eval()

        freeze_at = 5 - unlocked_groups
        print(f'Freeze the resnet at {freeze_at}', flush=True)

        if freeze_at >= 1:  # stem
            _lock(self.conv1)
            _lock(self.bn1)
            _lock(self.conv2)
            _lock(self.bn2)
            _lock(self.conv3)
            _lock(self.bn3)
        # each stage is a torch.nn.modules.container.Sequential
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=2):
            if freeze_at >= idx:
                for block in stage.children():  # each block is a Bottleneck
                    _lock(block)
        if self.freeze_all_bns:
            print(f'Freeze all bn layers', flush=True)           # TODO: study if this is necessary
            freeze_batch_norm_2d(self)

    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        with paddle.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes

    def extract_roi_features(self, x, normed_boxes, extract_type='v2'):
        if extract_type == 'v1':
            return self._extract_roi_features_v1(x, normed_boxes)
        else:
            assert extract_type == 'v2'
            return self._extract_roi_features_v2(x, normed_boxes)

    def mask_attn_pool(self, image, masks):
        return self.mask_pool(image, masks)

    def mask_pool(self, image, masks):
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature_map = self.attnpool.forward_dense(x)
        feature_map = F.normalize(feature_map, axis=1)          # remember to normalize!

        feature_map = feature_map.reshape((image.shape[0], -1, image.shape[2] * image.shape[3]))   # bs, c, h*w
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = paddle.concat(masks).astype('float32').reshape((image.shape[0], -1))    # bs, h*w
        feature_map = paddle.repeat(feature_map, num_masks_per_image, axis=0)
        features = (feature_map * masks.unsqueeze(1)).sum(axis=-1) / (masks.sum(axis=1, keepdim=True) + 1e-12)

        return features

    def _extract_roi_features_v1(self, x, normed_boxes, **kwargs):
        with paddle.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        x = self.layer4(x)

        x = self.attnpool.forward_dense(x)
        x = F.normalize(x, axis=1)          # remember to normalize!
        # TODO: debug
        roi_feats = roi_align(x, self._denormalize_boxes(normed_boxes, x),
                              (1, 1), 1.0, -1, True)[:, :, 0, 0]
        return roi_feats

    def _extract_roi_features_v2(self, x, normed_boxes, **kwargs):
        with paddle.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        x = self.layer4(x)    # only the last layer is finetuned in our implementation

        tar_size = self.attnpool_input_size
        # TODO: debug
        roi_feats = roi_align(x, self._denormalize_boxes(normed_boxes, x),
                              (tar_size, tar_size), 1.0, -1, True)

        roi_feats = self.attnpool(roi_feats)

        return roi_feats

    def encode_dense(self, x, keep_shape=True):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature_map = self.attnpool.forward_dense(x)
        feature_map = F.normalize(feature_map, axis=1)  # remember to normalize!

        return feature_map
