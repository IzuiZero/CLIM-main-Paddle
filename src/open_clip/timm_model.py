import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import roi_align
from collections import OrderedDict

try:
    import timm
    from timm.models.layers import Mlp, to_2tuple
    try:
        from timm.models.layers.attention_pool2d import RotAttentionPool2d
        from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
    except ImportError:
        from timm.layers import RotAttentionPool2d
        from timm.layers import AttentionPool2d as AbsAttentionPool2d
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d

class TimmModel(nn.Layer):
    """ timm model adapter """

    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool='avg',
            proj='linear',
            proj_bias=False,
            drop=0.,
            drop_path=None,
            patch_drop=None,
            pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = patch_drop

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if not proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            self.trunk = timm.create_model(
                model_name,
                num_classes=embed_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias_attr=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))
        else:
            assert not proj, f'Unknown projection type {proj}.'

        self.head = nn.Sequential(head_layers)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.trainable = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # Partial freeze requires latest timm (master) branch and is subject to change
            try:
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install the latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.parameters()[param].trainable = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

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

    def _extract_roi_features_v1(self, x, normed_boxes, **kwargs):
        h, w = x.shape[-2:]
        x = self.trunk.forward_features(x)
        h_f, w_f = x.shape[-2:]
        tar_h = (self.image_size[0] * h_f) // h
        tar_w = (self.image_size[1] * w_f) // w
        x = roi_align(x, self._denormalize_boxes(normed_boxes, x), (tar_h, tar_w),
                      1.0, -1, True)

        x = self.trunk.forward_head(x)
        x = self.head(x)

        return x

    def encode_dense(self, x, **kwargs):
        x = self.trunk.forward_features(x)
        x = self.dense_trunk_head(x)
        x = self.head(x)
        x = paddle.transpose(x, perm=[0, 3, 1, 2])

        return x

    def dense_trunk_head(self, x):
        x = self.trunk.head.norm(x)
        x = paddle.transpose(x, perm=[0, 2, 3, 1])
        x = self.trunk.head.drop(x)
        # x = paddle.transpose(x, perm=[0, 3, 1, 2])

        return x

    def mask_pool(self, image, masks):
        feature_map = self.encode_dense(image)
        feature_map = F.normalize(feature_map, axis=1)          # remember to normalize!
        feature_map = paddle.flatten(feature_map, start_axis=2, stop_axis=3)   # bs, c, h*w
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = paddle.concat(masks).astype('float32').flatten(1)    # bs, h*w
        feature_map = paddle.repeat(feature_map, repeats=num_masks_per_image, axis=0)
        features = paddle.sum(feature_map * paddle.unsqueeze(masks, axis=1), axis=-1) / (paddle.sum(masks, axis=1, keepdim=True) + 1e-12)

        return features

    def extract_roi_features(self, x, normed_boxes, extract_type='v1'):
        assert extract_type == "v1"
        if extract_type == 'v1':
            return self._extract_roi_features_v1(x, normed_boxes)
        else:
            assert extract_type == 'v2'
            return self._extract_roi_features_v2(x, normed_boxes)

    def _extract_roi_features_v2(self, x, normed_boxes, **kwargs):
        x = self.encode_dense(x)
        x = F.normalize(x, axis=1)  # remember to normalize!

        roi_feats = roi_align(x, self._denormalize_boxes(normed_boxes, x), (1, 1),
                              1.0, -1, True)[..., 0, 0]
        return roi_feats

    def encode_rois_and_image(self, x, normed_boxes, **kwargs):
        h, w = x.shape[-2:]
        x = self.trunk.forward_features(x)
        h_f, w_f = x.shape[-2:]
        tar_h = (self.image_size[0] * h_f) // h
        tar_w = (self.image_size[1] * w_f) // w
        x_image = x
        x_rois = roi_align(x, self._denormalize_boxes(normed_boxes, x), (tar_h, tar_w),
                           1.0, -1, True)

        x_rois = self.trunk.forward_head(x_rois)
        x_rois = self.head(x_rois)
        x_rois = F.normalize(x_rois, axis=-1)

        x_image = self.trunk.forward_head(x_image)
        x_image = self.head(x_image)
        x_image = F.normalize(x_image, axis=-1)

        return x_rois, x_image
