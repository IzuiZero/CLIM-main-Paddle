import open_clip
import paddle
from mmdet.registry import MODELS
from mmengine.model import BaseModule
import paddle.nn.functional as F


@MODELS.register_module()
class CLIPResNet(BaseModule):
    def __init__(self, model_name, cache_dir, pretrained='openai', roi_extractor=None):
        super().__init__()
        self.model_name = model_name
        clip_model = open_clip.create_model(model_name,
                                            pretrained=pretrained,
                                            cache_dir=cache_dir)
        self.visual = clip_model.visual
        self.roi_extractor = MODELS.build(roi_extractor)

    def init_weights(self):
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.stop_gradient = True

    def train(self, mode=True):
        self.training = mode
        self.visual.eval()  # set visual to eval mode
        return self

    def forward(self, x):
        outputs = []
        with paddle.no_grad():
            visual = self.visual
            x = visual.stem(x)
            for i in range(4):
                layer = getattr(visual, f'layer{i+1}')
                x = layer(x)
                outputs.append(x)

        return tuple(outputs)

    def clip_pool(self, clip_x, rois):
        roi_feats = self.roi_extractor([clip_x], rois)
        roi_feats = self.visual.attnpool(roi_feats)
        roi_feats = F.normalize(roi_feats, axis=-1)

        return roi_feats
