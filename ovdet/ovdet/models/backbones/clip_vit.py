import open_clip
import paddle
from paddle import nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
import paddle.nn.functional as F
from mmcv.cnn import build_norm_layer


@MODELS.register_module()
class CLIPViT(BaseModule):
    def __init__(self, model_name, cache_dir, pretrained='openai',
                 out_indices=[3, 5, 7, 11],
                 roi_extractor=None,
                 norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        clip_model = open_clip.create_model(model_name,
                                            pretrained=pretrained,
                                            cache_dir=cache_dir)
        self.embed_dim = embed_dim = clip_model.embed_dim  # output dim
        self.width = width = clip_model.visual.transformer.width
        self.patch_size = patch_size = clip_model.visual.patch_size[0]
        self.interpolate1 = nn.Sequential(
            nn.Conv2DTranspose(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.Conv2DTranspose(width, width, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Sequential(
            nn.Conv2DTranspose(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.visual = clip_model.visual

        self.roi_extractor = MODELS.build(roi_extractor)


    def init_weights(self):
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.stop_gradient = True

    def train(self, mode=True):
        self.training = mode
        self.visual.eval()  # set visual to eval mode
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self

    def forward(self, x):
        visual = self.visual
        bs, _, h, w = x.shape

        with paddle.no_grad():
            x = visual.conv1(x)  # shape = [*, width, grid, grid]
            bs, _, h, w = x.shape
            # assert h == w  # TODO: support input of any shape, need to change the normed boxes to real boxes
            x = x.reshape([x.shape[0], x.shape[1], -1])  # shape = [*, width, grid ** 2]
            x = x.transpose([0, 2, 1])  # shape = [*, grid ** 2, width]
            x = paddle.concat(
                [visual.class_embedding.cast(x.dtype) + paddle.zeros([x.shape[0], 1, x.shape[-1]], dtype=x.dtype),
                 x], axis=1)  # shape = [*, grid ** 2 + 1, width]
            if (h, w) == visual.grid_size:
                pe = visual.positional_embedding.cast(x.dtype)
            else:
                pe = visual.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

            x = x + pe

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            x = visual.patch_dropout(x)
            x = visual.ln_pre(x)

            outs = []
            x = x.transpose([1, 0, 2])  # NLD -> LND
            for i in range(visual.transformer.layers - 1):
                x = visual.transformer.resblocks[i](x)
                if i in self.vit_layers:
                    outs.append(self._expand_x(x, h, w))
            x = visual.transformer.resblocks[-1].forward_without_attn(x)
            if (visual.transformer.layers - 1) in self.vit_layers:
                outs.append(self._expand_x(x, h, w))

            if self.training:
                feature_map = None
            else:
                x = x.transpose([1, 0, 2])  # LND -> NLD
                if visual.attn_pool is not None:
                    x = visual.attn_pool(x)
                    x = visual.ln_post(x)
                    _, x = visual._global_pool(x)
                else:
                    _, x = visual._global_pool(x)
                    x = visual.ln_post(x)

                if visual.proj is not None:
                    x = paddle.matmul(x, visual.proj)

                feature_map = x.reshape([bs, h * w, -1])  # .transpose(0, 3, 1, 2)
                feature_map = F.normalize(feature_map, axis=-1)  # normalize at the last dimension
                feature_map = feature_map.reshape([bs, h, w, -1]).transpose([0, 3, 1, 2])

        assert len(outs) == 4
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        return tuple(outs)

    def _expand_x(self, x, h, w):
        # x: LND
        x = x[1:].transpose([1, 2, 0]).reshape([-1, self.width, h, w])

        return x

    def clip_pool(self, clip_x, rois):
        roi_feats = self.roi_extractor([clip_x], rois)[..., 0, 0]
        roi_feats = F.normalize(roi_feats, axis=-1)

        return roi_feats
